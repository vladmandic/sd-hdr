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
generator = None # torch generator
iterations = 3 # run dark/normal/light
timestep = 0 # from args.timestep
exp = 0.0 # from args.exp
iteration = 0 # counter
latent = None # saved latent
custom_timesteps = None # custom timesteps
total_steps = 0 # counter for total steps


def set_sampler(args):
    sampler = getattr(diffusers, args.sampler, None)
    if sampler is None:
        log.warning(f'Scheduler: sampler={args.sampler} invalid')
        log.info(f'Scheduler: current={pipe.scheduler.__class__.__name__}')
        return
    try:
        keys = inspect.signature(sampler, follow_wrapped=True).parameters.keys()
        config = {}
        for k, v in pipe.scheduler.config.items():
            if k in keys and not k.startswith('_'):
                config[k] = v
        pipe.scheduler = sampler.from_config(config)
        config = [{ k: v } for k, v in pipe.scheduler.config.items() if not k.startswith('_')]
        log.info(f'Scheduler: sampler={pipe.scheduler.__class__.__name__} config={config}')
    except Exception as e:
        log.error(f'Scheduler: {e}')
        log.info(f'Scheduler: current={pipe.scheduler.__class__.__name__}')


def patch():
    def retrieve_timesteps(scheduler, num_inference_steps, device, timesteps, sigmas, **kwargs): # pylint: disable=redefined-outer-name
        if custom_timesteps is None:
            return orig_retrieve_timesteps(scheduler, num_inference_steps, device, timesteps, sigmas, **kwargs)
        else:
            orig_retrieve_timesteps(scheduler, num_inference_steps, device, timesteps, sigmas, **kwargs) # run original
            return custom_timesteps, len(custom_timesteps) # but return reduced timesteps

    orig_retrieve_timesteps = diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.retrieve_timesteps
    diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.retrieve_timesteps = retrieve_timesteps


def load(args):
    global pipe, dtype, device, generator, exp, timestep, offload # pylint: disable=global-statement
    exp = args.exp
    timestep = args.timestep
    if args.dtype == 'fp16' or args.dtype == 'float16':
        dtype = torch.float16
    elif args.dtype == 'bf16' or args.dtype == 'bfloat16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    patch()
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
    if not args.model.lower().endswith('.safetensors'):
        args.model += '.safetensors'
    if not os.path.exists(args.model):
        log.error(f'Model: path="{args.model}" not found')
        return
    log.debug(f'Device: current={torch.cuda.current_device()} cuda={torch.cuda.is_available()} count={torch.cuda.device_count()} name="{torch.cuda.get_device_name(0)}"')
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
    if args.offload:
        pipe.enable_model_cpu_offload(device=device)
    t1 = time.time()
    log.info(f'Loaded: model="{args.model}" time={t1-t0:.2f}')
    log.debug(f'Memory: allocated={torch.cuda.memory_allocated() / 1e9:.2f} cached={torch.cuda.memory_reserved() / 1e9:.2f}')
    log.debug(f'Model: unet="{pipe.unet.dtype}/{pipe.unet.device}" vae="{pipe.vae.dtype}/{pipe.vae.device}" te1="{pipe.text_encoder.dtype}/{pipe.text_encoder.device}" te2="{pipe.text_encoder_2.device}/{pipe.text_encoder_2.device}"')
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
    return tokens, prompt_embeds.to(device), pooled_prompt_embeds.to(device)


def callback(p, step: int, ts: int, kwargs: dict): # pylint: disable=unused-argument
    def center_tensor(tensor, channel_shift=0.0, full_shift=0.0, offset=0.0):
        tensor -= tensor.mean(dim=(-1, -2), keepdim=True) * channel_shift
        tensor -= tensor.mean() * full_shift - offset
        return tensor

    def exp_correction(channel):
        channel[0:1] = center_tensor(channel[0:1], channel_shift=0.0, full_shift=1.0, offset=exp * (iteration -1) / 2)
        return channel

    global latent, total_steps # pylint: disable=global-statement
    total_steps += 1
    latents = kwargs.get('latents', None) # if we have latent stored, just use it and ignore what model returns
    if custom_timesteps is not None and latent is not None and ts == custom_timesteps[0]: # replace latent with stored one
        latents = latent.clone()
    if ts < timestep:
        if latent is None:
            latent = latents.clone() # store latent first time we get here
        for i in range(latents.shape[0]):
            latents[i] = exp_correction(latents[i])
    kwargs['latents'] = latents
    return kwargs


def decode(latents):
    image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    image = image.squeeze(0).permute(1, 2, 0)
    image = (image / 2 + 0.5).clamp(0, 1)
    image = (255 * image).float().cpu().numpy()
    image = image.astype(np.uint8)
    return image


def run(args, prompt):
    if pipe is None:
        log.error('Model: not loaded')
        return
    global iteration, latent, custom_timesteps, total_steps  # pylint: disable=global-statement
    torch.cuda.reset_peak_memory_stats()
    latent = None
    total_steps = 0
    tokens, embeds, pooled = encode_prompt(prompt)
    seed = args.seed if args.seed >= 0 else int(random.randrange(4294967294))
    custom_timesteps = None
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
        'output_type': 'latent',
        'return_dict': False,
        'callback_on_step_end': callback,
    }
    with torch.inference_mode():
        ts = int(time.time())
        images = []
        t0 = time.time()

        for i in range(iterations):
            iteration = i
            t1 = time.time()
            generator.manual_seed(seed)
            latents = pipe(**kwargs)[0]
            custom_timesteps = pipe.scheduler.timesteps.clone()
            custom_timesteps = custom_timesteps[custom_timesteps < timestep] # only use timesteps below ts threshold for future runs
            image = decode(latents)
            images.append(image)
            t2 = time.time()
            if args.save:
                name = os.path.join(args.output, f'{ts}-{i}.png')
                cv2.imwrite(name, image)
                log.debug(f'Image: i={iteration+1}/{iterations} seed={seed} shape={image.shape} name="{name}" time={t2-t1:.2f}')

        if args.ldr or args.hdr:
            try:
                align = cv2.createAlignMTB()
                aligned_images = []
                for img in images:
                    aligned = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    align.process([img], [aligned])
                    aligned_images.append(aligned)
                log.debug(f'OpenCV: aligned={[img.shape for img in aligned_images]}')
                merge = cv2.createMergeMertens()
                hdr = merge.process(aligned_images)
                ldr = np.clip(hdr * 255, 0, 255).astype(np.uint8)
                hdr = np.clip(hdr * 65535, 0, 65535).astype(np.uint16)
                its = len(images) * total_steps / (t2 - t0)
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
                log.info(f'Merge: seed={seed} hdr="{name_hdr}" ldr="{name_ldr}" json="{name_json}" time={t2-t0:.2f} total-steps={total_steps} its={its:.2f}')
            except cv2.error as e:
                log.error(f'OpenCV: shapes={[img.shape for img in images]} dtypes={[img.dtype for img in images]} {e}')
                raise
    mem = dict(torch.cuda.memory_stats())
    log.debug(f'Memory: peak={mem["active_bytes.all.peak"]} retries={mem["num_alloc_retries"]} oom={mem["num_ooms"]}')
