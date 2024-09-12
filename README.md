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

> ./hdr.sh --model /mnt/sdxl/TempestV0.1-Artistic.safetensors --prompts prompts.txt --output tmp/ --save --hdr --ldr

```log
10:04:00-175852 INFO     HDR start
10:04:00-177997 INFO     Env: python=3.12.3 platform=Linux bin="/home/vlado/dev/sd-hdr/venv/bin/python" venv="/home/vlado/dev/sd-hdr/venv"
10:04:00-178744 INFO     Args: Namespace(dtype='bfloat16', device='cuda:0', model='sdxl/TempestV0.1-Artistic.safetensors', width=1024, height=1024, steps=20, seed=-1, cfg=7.0, sampler='EulerAncestralDiscreteScheduler', prompts='prompts.txt', output='tmp', exp=1.0, timestep=200, save=True, hdr=True, ldr=True, json=False, debug=True)
10:04:00-327286 DEBUG    Device: current=0 cuda=True count=1 name="NVIDIA GeForce RTX 4090"
10:04:00-328066 INFO     Loading: model="sdxl/TempestV0.1-Artistic.safetensors" dtype="torch.bfloat16" device="cuda:0"
10:04:03-339978 INFO     Loaded: model="sdxl/TempestV0.1-Artistic.safetensors" time=3.01
10:04:03-341036 DEBUG    Memory: allocated=8.39 cached=8.66
10:04:03-343692 DEBUG    Model: unet="torch.bfloat16/cuda:0" vae="torch.bfloat16/cuda:0" te1="torch.bfloat16/cuda:0" te2="cuda:0/cuda:0"
10:04:03-345122 INFO     Scheduler: sampler=EulerAncestralDiscreteScheduler config=[{'num_train_timesteps': 1000}, {'beta_start': 0.00085}, {'beta_end': 0.012}, {'beta_schedule': 'scaled_linear'}, {'trained_betas': None}, {'prediction_type': 'epsilon'}, {'timestep_spacing': 'leading'}, {'steps_offset': 1}, {'rescale_betas_zero_snr': False}]
10:04:03-345956 INFO     Sequence: count=1/2
10:04:03-455445 INFO     Generate: prompt="cute robot walking on a surface of a lake in a tropical forrest" tokens=15 seed=2040646293
10:04:25-107072 DEBUG    Image: i=1/3 seed=2040646293 shape=(1024, 1024, 3) name="tmp/1726149843-0.png" time=21.62
10:04:25-748966 DEBUG    Image: i=2/3 seed=2040646293 shape=(1024, 1024, 3) name="tmp/1726149843-1.png" time=0.61
10:04:26-421777 DEBUG    Image: i=3/3 seed=2040646293 shape=(1024, 1024, 3) name="tmp/1726149843-2.png" time=0.64
10:04:26-446687 DEBUG    OpenCV: aligned=[(1024, 1024, 3), (1024, 1024, 3), (1024, 1024, 3)]
10:04:26-583357 INFO     Merge: seed=2040646293 hdr="tmp/1726149843-hdr.png" ldr="tmp/1726149843-ldr.png" json="None" time=22.94 total-steps=28 its=3.66
10:04:26-584285 INFO     Sequence: count=2/2
10:04:26-601033 INFO     Generate: prompt="beautiful woman in a long dress walking through a busy city street during sunset" tokens=16 seed=1676172106
10:04:29-141769 DEBUG    Image: i=1/3 seed=1676172106 shape=(1024, 1024, 3) name="tmp/1726149866-0.png" time=2.51
10:04:29-788513 DEBUG    Image: i=2/3 seed=1676172106 shape=(1024, 1024, 3) name="tmp/1726149866-1.png" time=0.62
10:04:30-465556 DEBUG    Image: i=3/3 seed=1676172106 shape=(1024, 1024, 3) name="tmp/1726149866-2.png" time=0.65
10:04:30-488897 DEBUG    OpenCV: aligned=[(1024, 1024, 3), (1024, 1024, 3), (1024, 1024, 3)]
10:04:30-621299 INFO     Merge: seed=1676172106 hdr="tmp/1726149866-hdr.png" ldr="tmp/1726149866-ldr.png" json="None" time=3.83 total-steps=28 its=21.91
10:04:30-622190 INFO     HDR end
```

## Note

- A lot of optimizations are possible, this is just a quick and dirty script to get started  
- Prompts file should have one prompt per line  
- Created filename is simple epoch timestamp
