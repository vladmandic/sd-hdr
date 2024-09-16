# StableDiffusion HDR

*Generate HDR images using StableDiffusionXL*

## Example

![example-hdr](https://github.com/user-attachments/assets/3c17e4d9-31d3-4983-8c5f-d4f23c44fd3f)

## Run

```js
$ ./hdr.sh --help

options:
  --dtype DTYPE         torch dtype
  --device DEVICE       torch device
  --model MODEL         sd model
  --width WIDTH         image width
  --height HEIGHT       image height
  --steps STEPS         sampling steps
  --seed SEED           noise seed
  --cfg CFG             cfg scale
  --sampler SAMPLER     sd sampler
  --prompts PROMPTS     prompts file
  --output OUTPUT       output folder
  --format {png,hdr,dng,tiff,all}
                        hdr file format
  --exp EXP             exposure correction
  --timestep TIMESTEP   correction timestep
  --save                save interim images
  --ldr                 create 8bpc hdr png image
  --json                save params to json
  --debug               debug log
  --offload             offload model components
```

> ./hdr.sh --model /mnt/sdxl/TempestV0.1-Artistic.safetensors --prompts prompts.txt --output tmp/ --save --format:tiff --ldr --json --debug

```log
09:49:12-373969 INFO     HDR start
09:49:12-375916 INFO     Env: python=3.12.3 platform=Linux bin="/home/vlado/dev/sd-hdr/venv/bin/python" venv="/home/vlado/dev/sd-hdr/venv"
09:49:12-376511 INFO     Args: Namespace(dtype='bfloat16', device='auto', model='/mnt/models/stable-diffusion/sdxl/TempestV0.1-Artistic.safetensors', width=800, height=600, steps=20, seed=-1, cfg=7.0, sampler='EulerAncestralDiscreteScheduler', prompts='prompts.txt', output='tmp', format='all', exp=1.0, timestep=200, save=False, ldr=True, json=True, debug=True, offload=False)
09:49:13-945383 DEBUG    Device: current=0 cuda=True count=1 name="NVIDIA GeForce RTX 4090"
09:49:13-946320 INFO     Loading: model="/mnt/models/stable-diffusion/sdxl/TempestV0.1-Artistic.safetensors" dtype="torch.bfloat16" device="cuda"
09:49:16-958526 INFO     Loaded: model="/mnt/models/stable-diffusion/sdxl/TempestV0.1-Artistic.safetensors" time=3.01
09:49:16-959421 DEBUG    Memory: allocated=8.40 cached=8.65
09:49:16-962064 DEBUG    Model: unet="torch.bfloat16/cuda:0" vae="torch.bfloat16/cuda:0" te1="torch.bfloat16/cuda:0" te2="cuda:0/cuda:0"
09:49:16-963839 INFO     Scheduler: sampler=EulerAncestralDiscreteScheduler config=[{'num_train_timesteps': 1000}, {'beta_start': 0.00085}, {'beta_end': 0.012}, {'beta_schedule': 'scaled_linear'}, {'trained_betas': None}, {'prediction_type': 'epsilon'}, {'timestep_spacing': 'leading'}, {'steps_offset': 1}, {'rescale_betas_zero_snr': False}]
09:49:16-964700 INFO     Sequence: count=1/2
09:49:17-110308 INFO     Generate: prompt="cute robot walking on a surface of a lake in a tropical forrest during sunrise" tokens=17 seed=2291510946
09:49:28-421951 INFO     Save: type=ldr format=png file="tmp/1726494557-ldr.png"
09:49:28-466720 INFO     Save: type=hdr format=tif file="tmp/1726494557.tiff"
09:49:28-508290 INFO     Save: type=json file="tmp/1726494557.json"
09:49:28-508999 INFO     Merge: seed=2291510946 format="all" time=11.15 total-steps=28 its=7.53
09:49:28-509686 DEBUG    Stats: hdr={'kurtosis': -0.82436203956604, 'msd': 0.07170082628726959, 'range': 199.99986267089844, 'entropy': 15.402483940124512} ldr={'kurtosis': -0.8215048313140869, 'msd': 1.0867865967156831e-06, 'range': 151.80120849609375, 'entropy': 7.774331092834473}
09:49:28-510480 DEBUG    Memory: peak=17.51 retries=0 oom=0
09:49:28-511019 INFO     Sequence: count=2/2
09:49:28-526896 INFO     Generate: prompt="beautiful woman in a long dress walking through a busy city street during sunset with rain falling" tokens=19 seed=1997992512
09:49:30-709413 INFO     Save: type=ldr format=png file="tmp/1726494568-ldr.png"
09:49:30-757896 INFO     Save: type=hdr format=tif file="tmp/1726494568.tiff"
09:49:30-804970 INFO     Save: type=json file="tmp/1726494568.json"
09:49:30-805658 INFO     Merge: seed=1997992512 format="all" time=2.00 total-steps=28 its=41.96
09:49:30-806203 DEBUG    Stats: hdr={'kurtosis': -1.1626012325286865, 'msd': 0.07364531606435776, 'range': 199.99986267089844, 'entropy': 15.6673583984375} ldr={'kurtosis': -1.1612789630889893, 'msd': 1.1155688071085024e-06, 'range': 151.80120849609375, 'entropy': 7.8487467765808105}
09:49:30-806945 DEBUG    Memory: peak=9.58 retries=0 oom=0
09:49:30-807481 INFO     HDR end
```

## Note

- A lot of optimizations are possible, this is just a quick and dirty script to get started  
- Prompts file should have one prompt per line  
- Created filename is simple epoch timestamp
- Output formats:
  - Pass: png/hdr/tiff
  - Fail: exr/dng
