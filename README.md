# StableDiffusion HDR

*Generate HDR images using StableDiffusionXL*

## Examples

## Run

```js
$ ./hdr.sh --help

options:
  --dtype DTYPE        torch dtype
  --device DEVICE      torch device
  --model MODEL        sd model
  --width WIDTH        image width
  --height HEIGHT      image height
  --steps STEPS        sampling steps
  --seed SEED          noise seed
  --cfg CFG            cfg scale
  --sampler SAMPLER    sd sampler
  --prompts PROMPTS    prompts file
  --output OUTPUT      output folder
  --exp EXP            exposure correction
  --timestep TIMESTEP  correction timestep
  --save               save interim images
  --hdr                create 16bpc hdr image
  --ldr                create 8bpc hdr image
  --json               save params to json
  --debug              debug log
```

> ./hdr.sh --model /mnt/sdxl/TempestV0.1-Artistic.safetensors --prompts prompts.txt --output tmp/ --save --hdr --ldr

```log
19:22:38-716570 INFO     Env: python=3.12.3 platform=Linux bin="/home/vlado/dev/sd-hdr/venv/bin/python3" venv="/home/vlado/dev/sd-hdr/venv"
19:22:38-717295 INFO     Args: Namespace(dtype='bfloat16', device='cuda:0', model='/mnt/sdxl/TempestV0.1-Artistic.safetensors', width=1024, height=1024, steps=10, seed=-1, cfg=7.0, sampler='UniPCMultistepScheduler', prompts='prompts.txt', output='tmp/', exp=1.0, timestep=200, save=True, hdr=True, ldr=True, json=True, debug=False)
19:22:38-816148 INFO     Loading: model="/mnt/sdxl/TempestV0.1-Artistic.safetensors" dtype="torch.bfloat16" device="cuda:0"
19:22:48-995316 INFO     Loaded: model="/mnt/sdxl/TempestV0.1-Artistic.safetensors" time=10.18
19:22:48-998194 INFO     Sampler: <class 'diffusers.schedulers.scheduling_unipc_multistep.UniPCMultistepScheduler'> config=[...]
19:22:48-999942 INFO     Sequence: count=1/3
19:22:49-197405 INFO     Generate: prompt="cute robot" tokens=4 seed=1787489337
19:23:12-404111 INFO     Merge: seed=1787489337 hdr="tmp/1726010569-hdr.png" ldr="tmp/1726010569-ldr.png" json="tmp/1726010569.json" time=23.02 its=1.30
19:23:12-404988 INFO     Sequence: count=2/3
19:23:12-419093 INFO     Generate: prompt="airplane in the sky" tokens=6 seed=858719668
19:23:16-612599 INFO     Merge: seed=858719668 hdr="tmp/1726010592-hdr.png" ldr="tmp/1726010592-ldr.png" json="tmp/1726010592.json" time=4.02 its=7.46
19:23:16-613379 INFO     Sequence: count=3/3
19:23:16-626820 INFO     Generate: prompt="a big tree" tokens=5 seed=816301649
19:23:20-830487 INFO     Merge: seed=816301649 hdr="tmp/1726010596-hdr.png" ldr="tmp/1726010596-ldr.png" json="tmp/1726010596.json" time=4.01 its=7.48
```
