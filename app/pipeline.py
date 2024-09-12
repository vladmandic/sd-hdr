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

    if sampler is None:
        log.warning(f"Invalid sampler: {args.sampler}. The current scheduler will be kept.")
        log.info(f"Current scheduler: {pipe.scheduler.__class__.__name__}")
        return

    try:
        keys = inspect.signature(sampler, follow_wrapped=True).parameters.keys()
        config = {}
        for k, v in pipe.scheduler.config.items():
            if k in keys and not k.startswith('_'):
                config[k] = v
        pipe.scheduler = sampler.from_config(config)
        log.info(f'Sampler successfully set: {pipe.scheduler.__class__.__name__}')
        config = [{ k: v } for k, v in pipe.scheduler.config.items() if not k.startswith('_')]
        log.info(f'Sampler config: {config}')
    except Exception as e:
        log.error(f"Error setting sampler: {e}. The current scheduler will be kept.")
        log.info(f"Current scheduler: {pipe.scheduler.__class__.__name__}")


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
    if iteration > 0:
        for i in range(latents.shape[0]):
            latents[i] = exp_correction(latents[i])
    kwargs['latents'] = latents
    return kwargs


def run(args, prompt):
    global iteration, timestep # pylint: disable=global-statement

    # Debug information about GPU usage
    log.debug(f"CUDA available: {torch.cuda.is_available()}")
    log.debug(f"Current device: {torch.cuda.current_device()}")
    log.debug(f"Device count: {torch.cuda.device_count()}")
    log.debug(f"Device name: {torch.cuda.get_device_name(0)}")
    log.debug(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    log.debug(f"Memory cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    # Check if the pipe is on the correct device
    log.debug(f"UNet device: {pipe.unet.device}")
    log.debug(f"VAE device: {pipe.vae.device}")
    log.debug(f"Text Encoder device: {pipe.text_encoder.device}")

    tokens, embeds, pooled = encode_prompt(prompt)
    seed = args.seed if args.seed >= 0 else int(random.randrange(4294967294))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    generator.manual_seed(seed)

    log.info(f'Generate: prompt="{prompt}" tokens={len(tokens)} seed={seed}')
    
    hdr_target_step = 2
    save_step = args.steps - hdr_target_step

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

        # First full generation
        # Modify the pipeline to capture intermediate latents
        original_step_function = pipe.scheduler.step
        latents_history = []

        def step_with_capture(model_output, timestep, sample, **kwargs):
            latents_history.append(sample.clone())
            return original_step_function(model_output, timestep, sample, **kwargs)

        pipe.scheduler.step = step_with_capture

        latents = pipe(**kwargs)[0]
        
        # Restore original step function
        pipe.scheduler.step = original_step_function

        latents_at_save_step = latents_history[save_step]
        original_timesteps = pipe.scheduler.timesteps.clone()

        for i in range(3):
            iteration = i
            t1 = time.time()

            if i == 0:
                current_latents = latents
            else:
                current_latents = latents_at_save_step.clone()
                # Set the correct timestep when resuming
                timestep = original_timesteps[save_step]
                pipe.scheduler._step_index = save_step
                # Run only the last steps
                current_kwargs = kwargs.copy()
                current_kwargs['num_inference_steps'] = hdr_target_step
                current_kwargs['latents'] = current_latents
                current_latents = pipe(**current_kwargs)[0]

            # Decode latents to image
            image = pipe.vae.decode(current_latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
            log.debug(f"Shape after VAE decode: {image.shape}")

            image = image.squeeze(0).permute(1, 2, 0)
            log.debug(f"Shape after permute: {image.shape}")

            image = (image / 2 + 0.5).clamp(0, 1)
            image = (255 * image).float().cpu().numpy()
            log.debug(f"Shape after numpy conversion: {image.shape}")

            image = image.astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            log.debug(f"Final image shape: {image.shape}")
            
            images.append(image)

            t2 = time.time()

            if args.save:
                name = os.path.join(args.output, f'{ts}-{i}.png')
                cv2.imwrite(name, image)
                its = args.steps / (t2 - t1)
                log.debug(f'Image: i={iteration+1}/{iterations} seed={seed} shape={image.shape} name="{name}" time={t2-t1:.2f} its={its:.2f}')

        if args.ldr or args.hdr:
            try:
                align = cv2.createAlignMTB()
                aligned_images = []
                for img in images:
                    aligned = img.copy()
                    align.process([img], [aligned])
                    aligned_images.append(aligned)
                log.debug(f"Aligned image shapes: {[img.shape for img in aligned_images]}")
                merge = cv2.createMergeMertens()
                hdr = merge.process(aligned_images)
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
            except cv2.error as e:
                log.error(f"OpenCV error during alignment or merging: {str(e)}")
                log.debug(f"Image shapes: {[img.shape for img in images]}")
                log.debug(f"Image dtypes: {[img.dtype for img in images]}")
                raise
