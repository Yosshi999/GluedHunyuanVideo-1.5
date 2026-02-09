# Glued Hunyuan Video 1.5
Glue the boundaries of latents to enable looped video generation of Diffuser-based Hunyuan Video 1.5.

## System Requirements
* Hardware: NVIDIA GPU, 14GB at minimumm
* Software: Python >= 3.10, Diffusers >= 0.36.0, and PyTorch

See [the original repository](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5?tab=readme-ov-file#-system-requirements) for detail.

## Installation
```
git clone https://github.com/Yosshi999/GluedHunyuanVideo-1.5
cd GluedHunyuanVideo-1.5
pip install .
```

## Quick Start
See [Usage with Diffusers](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5?tab=readme-ov-file#usage-with-diffusers) for usage of Hunyuan itself.

```python
import torch
from diffusers import HunyuanVideo15Pipeline
from diffusers.utils import export_to_video
from glue_hyvideo import patch_pipeline

dtype = torch.bfloat16
device = "cuda:0"
seed = 42
prompt = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard, and moves from right to left viewed from the camera. The scene is in an animated style."

pipe = HunyuanVideo15Pipeline.from_pretrained("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v", torch_dtype=dtype)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

pipe = patch_pipeline(pipe, glued_dims=(True, False, True))

generator = torch.Generator(device=device).manual_seed(seed)

video = pipe(
    prompt=prompt,
    generator=generator,
    num_frames=121,
    num_inference_steps=50,
).frames[0]

export_to_video(video, "output.mp4", fps=24)
```

## Known bugs
* PyTorch == 2.9.0 causes OOM in processing Conv3D of the VAE decoder. Upgrade to 2.10 or nightly.

# Notice
Tencent Hunyuan is licensed under the Tencent Hunyuan Community License Agreement, Copyright © 2025 Tencent. All Rights Reserved. The trademark rights of “Tencent Hunyuan” are owned by Tencent or its affiliate.