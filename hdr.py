#!/usr/bin/env python

import os
import sys
import logging
import platform
import argparse
from app.logger import log


def load_list(f):
    ext = os.path.splitext(f)[1].lower()
    if f is None or f == '':
        return []
    elif os.path.exists(f) and ext not in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.hdr', '.webp']:
        with open(f, 'r', encoding='utf8') as f:
            return [line.strip() for line in f.readlines()]
    else:
        return [f] # use as string


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HDR')
    parser.add_argument('--dtype', type=str, default='bfloat16', help='torch dtype')
    parser.add_argument('--device', type=str, default='auto', help='torch device')
    parser.add_argument('--model', type=str, required=True, help='sd model')
    parser.add_argument('--width', type=int, default=0, help='image width')
    parser.add_argument('--height', type=int, default=0, help='image height')
    parser.add_argument('--steps', type=int, default=20, help='sampling steps')
    parser.add_argument('--seed', type=int, default=-1, help='noise seed')
    parser.add_argument('--cfg', type=float, default=7.0, help='cfg scale')
    parser.add_argument('--sampler', type=str, default='EulerAncestralDiscreteScheduler', help='sd sampler')
    parser.add_argument('--prompt', type=str, required=True, help='prompt or prompts file')
    parser.add_argument('--negative', type=str, default="", help="negative prompt or prompts file")
    parser.add_argument('--image', type=str, default="", help="init image(s)")
    parser.add_argument('--strength', type=float, default=0.6, help='denoise strength')
    parser.add_argument('--output', type=str, required=True, help='output folder')
    parser.add_argument('--format', type=str, required=False, default='all', choices=['png', 'hdr', 'dng', 'tiff', 'all'], help='hdr file format')
    parser.add_argument('--exp', type=float, default=1.0, help='exposure correction')
    parser.add_argument('--timestep', type=int, default=200, help='correction timestep')
    parser.add_argument('--save', action='store_true', help='save interim images')
    parser.add_argument('--ldr', action='store_true', help='create 8bpc hdr png image')
    parser.add_argument('--json', action='store_true', help='save params to json')
    parser.add_argument('--debug', action='store_true', help='debug log')
    parser.add_argument('--offload', action='store_true', help='offload model components')
    args = parser.parse_args()
    if args.debug:
        log.setLevel(logging.DEBUG)
    log.info('HDR start')
    log.info(f'Env: python={platform.python_version()} platform={platform.system()} bin="{sys.executable}" venv="{sys.prefix}"')
    log.info(f'Args: {args}')
    os.makedirs(args.output, exist_ok=True)
    os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

    import app.pipeline
    prompts = load_list(args.prompt)
    negatives = load_list(args.negative)
    images = load_list(args.image)
    log.info(f'Inputs: prompts={len(prompts)} negatives={len(negatives)} images={len(images)}')
    app.pipeline.load(args, img2img = len(images) > 0)
    for i, prompt in enumerate(prompts):
        if len(prompt) == 0:
            continue
        negative = negatives[i % len(negatives)] if len(negatives) > 0 else ""
        image = images[i % len(images)] if len(images) > 0 else ""
        log.info(f'Sequence: count={i+1}/{len(prompts)}')
        app.pipeline.run(args, prompt, negative, image)
    log.info('HDR end')
