import os
import time
import json
import random
import inspect
import cv2
import numpy as np
import torch
import diffusers
from app.logger import log


pipe: diffusers.StableDiffusionXLPipeline = None
dtype = None
device = None
generator = None
iteration = 0
iterations = 3
timestep = 0
exp = 0.0


def set_sampler(args):
    sampler = getattr(diffusers, args.sampler, None)
    keys = inspect.signature(sampler, follow_wrapped=True).parameters.keys()
    config = {}
    for k, v in pipe.scheduler.config.items():
        if k in keys and not k.startswith('_'):
            config[k] = v
    pipe.scheduler = sampler.from_config(config)
    config = [{ k: v } for k, v in pipe.scheduler.config.items() if not k.startswith('_')]
    log.info(f'Sampler: {pipe.scheduler.__class__} config={config}')


def load(args):
    global pipe, dtype, device, generator, exp, timestep # pylint: disable=global-statement
    exp = args.exp
    timestep = args.timestep
    if args.dtype == 'fp16' or args.dtype == 'float16':
        dtype = torch.float16
    elif args.dtype == 'bf16' or args.dtype == 'bfloat16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    torch.set_default_dtype(dtype)
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark_limit = 0
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cudnn.deterministic = True
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    device = torch.device(args.device)
    generator = torch.Generator(device=device)
    log.info(f'Loading: model="{args.model}" dtype="{dtype}" device="{device}"')
    t0 = time.time()
    kwargs = {
        'torch_dtype': dtype,
        'safety_checker': None,
        'low_cpu_mem_usage': True,
        'use_safetensors': True,
        'add_watermarker': False,
        'force_upcast': False
    }
    pipe = diffusers.StableDiffusionXLPipeline.from_single_file(args.model, **kwargs).to(dtype=dtype, device=device)
    pipe.set_progress_bar_config(disable=True)
    pipe.fuse_qkv_projections()
    pipe.unet.eval()
    pipe.vae.eval()
    t1 = time.time()
    log.info(f'Loaded: model="{args.model}" time={t1-t0:.2f}')
    set_sampler(args)


def encode_prompt(prompt):
    tokens = pipe.tokenizer(prompt)['input_ids']
    (
        prompt_embeds,
        _negative_prompt_embeds,
        pooled_prompt_embeds,
        _negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        clip_skip=0,
    )
    return tokens, prompt_embeds, pooled_prompt_embeds


def callback(p, step: int, ts: int, kwargs: dict): # pylint: disable=unused-argument
    def center_tensor(tensor, channel_shift=0.0, full_shift=0.0, offset=0.0):
        tensor -= tensor.mean(dim=(-1, -2), keepdim=True) * channel_shift
        tensor -= tensor.mean() * full_shift - offset
        return tensor

    def exp_correction(channel):
        channel[0:1] = center_tensor(channel[0:1], channel_shift=0.0, full_shift=1.0, offset=exp * (iteration -1) / 2)
        return channel

    latents = kwargs.get('latents', None)
    if latents is not None and ts < timestep:
        for i in range(latents.shape[0]):
            latents[i] = exp_correction(latents[i])
    kwargs['latents'] = latents
    return kwargs


def run(args, prompt):
    global iteration # pylint: disable=global-statement
    tokens, embeds, pooled = encode_prompt(prompt)
    seed = args.seed if args.seed >= 0 else int(random.randrange(4294967294))
    log.info(f'Generate: prompt="{prompt}" tokens={len(tokens)} seed={seed}')
    kwargs = {
        'width': args.width,
        'height': args.height,
        'prompt_embeds': embeds,
        'pooled_prompt_embeds': pooled,
        'guidance_scale': args.cfg,
        'num_inference_steps': args.steps,
        'num_images_per_prompt': 1,
        'generator': generator,
        'output_type': 'np',
        'return_dict': False,
        'callback_on_step_end': callback,
    }
    with torch.inference_mode():
        ts = int(time.time())
        images = []
        t0 = time.time()
        for i in range(3): # TODO use set_timesteps to run only last two steps instead of entire sequence
            iteration = i
            generator.manual_seed(seed)
            t1 = time.time()
            image = pipe(**kwargs)[0][0]
            t2 = time.time()
            image = (255 * image).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
            if args.save:
                name = os.path.join(args.output, f'{ts}-{i}.png')
                cv2.imwrite(name, image)
                its = args.steps / (t2 - t1)
                log.debug(f'Image: i={iteration+1}/{iterations} seed={seed} shape={image.shape} name="{name}" time={t2-t1:.2f} its={its:.2f}')
        if args.ldr or args.hdr:
            align = cv2.createAlignMTB()
            align.process(images, images)
            merge = cv2.createMergeMertens()
            hdr = merge.process(images)
            ldr = np.clip(hdr * 255, 0, 255).astype(np.uint8)
            hdr = np.clip(hdr * 65535, 0, 65535).astype(np.uint16)
            its = len(images) * args.steps / (t2 - t0)
            name_ldr = None
            name_hdr = None
            name_json = None
            if args.ldr:
                name_ldr = os.path.join(args.output, f'{ts}-ldr.png')
                cv2.imwrite(name_ldr, ldr)
            if args.hdr:
                name_hdr = os.path.join(args.output, f'{ts}-hdr.png')
                cv2.imwrite(name_hdr, hdr)
            if args.json:
                name_json = os.path.join(args.output, f'{ts}.json')
                dct = args.__dict__.copy()
                dct['prompt'] = prompt
                dct['seed'] = seed
                json.dumps(dct, indent=4)
                with open(name_json, 'w', encoding='utf8') as f:
                    f.write(json.dumps(dct, indent=4))
            log.info(f'Merge: seed={seed} hdr="{name_hdr}" ldr="{name_ldr}" json="{name_json}" time={t2-t0:.2f} its={its:.2f}')
