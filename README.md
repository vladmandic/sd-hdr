# StableDiffusion HDR

*Generate HDR images using StableDiffusionXL*

## Example

![example-hdr](https://github.com/user-attachments/assets/3c17e4d9-31d3-4983-8c5f-d4f23c44fd3f)

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

> ./hdr.sh --model /mnt/sdxl/TempestV0.1-Artistic.safetensors --prompts prompts.txt --output tmp/ --save --hdr --ldr --json --debug

```log
13:43:01-217948 INFO     Env: python=3.12.3 platform=Linux bin="/home/vlado/dev/sd-hdr/venv/bin/python" venv="/home/vlado/dev/sd-hdr/venv"
13:43:01-218585 INFO     Args: Namespace(dtype='bfloat16', device='cuda:0', model='/mnt/models/stable-diffusion/sdxl/TempestV0.1-Artistic.safetensors', width=1024, height=1024, steps=20, seed=-1, cfg=7.0, sampler='EulerAncestralDiscreteScheduler', prompts='prompts.txt', output='tmp', exp=1.0, timestep=200, save=True, hdr=True, ldr=True, json=True, debug=True, offload=False)
13:43:01-411016 DEBUG    Device: current=0 cuda=True count=1 name="NVIDIA GeForce RTX 4090"
13:43:01-411742 INFO     Loading: model="/mnt/models/stable-diffusion/sdxl/TempestV0.1-Artistic.safetensors" dtype="torch.bfloat16" device="cuda:0"
13:43:04-486064 DEBUG    Memory: allocated=8.39 cached=8.66
13:43:04-488902 DEBUG    Model: unet="torch.bfloat16/cuda:0" vae="torch.bfloat16/cuda:0" te1="torch.bfloat16/cuda:0" te2="cuda:0/cuda:0"
13:43:04-490323 INFO     Scheduler: sampler=EulerAncestralDiscreteScheduler config=[...]
13:43:04-491242 INFO     Sequence: count=1/1
13:43:04-630333 INFO     Generate: prompt="cute robot walking on a surface of a lake in a tropical forrest during sunrise" tokens=17 seed=4260888451
13:43:26-396197 DEBUG    Image: i=1/3 seed=4260888451 shape=(1024, 1024, 3) name="tmp/1726162984-0.png" time=21.64 stats={'kurtosis': -0.46951746940612793, 'msd': 1.0425526397739304e-06, 'range': 151.80120849609375, 'entropy': 7.75035285949707}
13:43:27-177786 DEBUG    Image: i=2/3 seed=4260888451 shape=(1024, 1024, 3) name="tmp/1726162984-1.png" time=0.65 stats={'kurtosis': -0.8761804103851318, 'msd': 1.044339910549752e-06, 'range': 151.80120849609375, 'entropy': 7.835085868835449}
13:43:27-965057 DEBUG    Image: i=3/3 seed=4260888451 shape=(1024, 1024, 3) name="tmp/1726162984-2.png" time=0.66 stats={'kurtosis': -0.9673399925231934, 'msd': 9.421548270438507e-07, 'range': 151.80120849609375, 'entropy': 7.705101013183594}
13:43:28-379984 INFO     Merge: seed=4260888451 hdr="tmp/1726162984-hdr.png" ldr="tmp/1726162984-ldr.png" json="tmp/1726162984.json" time=23.20 total-steps=28 its=3.62
13:43:28-380780 DEBUG    Stats: hdr={'kurtosis': -0.7930150032043457, 'msd': 0.06874074786901474, 'range': 199.99986267089844, 'entropy': 15.490011215209961} ldr={'kurtosis': -0.7899765968322754, 'msd': 1.0418782494525658e-06, 'range': 151.80120849609375, 'entropy': 7.807065963745117}
13:43:33-204780 DEBUG    Memory: peak=9.88 retries=0 oom=0
```

## Note

- A lot of optimizations are possible, this is just a quick and dirty script to get started  
- Prompts file should have one prompt per line  
- Created filename is simple epoch timestamp
