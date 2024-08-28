# StableNormal

[![Website](https://raw.githubusercontent.com/prs-eth/Marigold/main/doc/badges/badge-website.svg)](https://stable-x.github.io/StableNormal)
[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2406.16864)
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face%20-Space-yellow)](https://huggingface.co/spaces/Stable-X/StableNormal)
[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face%20-Model-green)](https://huggingface.co/Stable-X/stable-normal-v0-1)
[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)

We propose StableNormal, which tailors the diffusion priors for monocular normal estimation. Unlike prior diffusion-based works, we focus on enhancing estimation stability by reducing the inherent stochasticity of diffusion models ( i.e. , Stable Diffusion). This enables “Stable-and-Sharp” normal estimation, which outperforms multiple baselines (try [Compare](https://huggingface.co/spaces/Stable-X/normal-estimation-arena)), and improves various real-world applications (try [Demo](https://huggingface.co/spaces/Stable-X/StableNormal)). 

![teaser](doc/StableNormal-Teaser.jpg)

## News
### 🎉 New Release: StableDelight 🎉
We're excited to announce the release of StableDelight, our latest open-source project focusing on real-time reflection removal from textured surfaces. Check out the [StableDelight](https://github.com/Stable-X/StableDelight) for more details!
![image](https://github.com/user-attachments/assets/fb138d2a-3fb4-4b86-ba51-3a60b91c8caf)

## Installation:

Please run following commands to build package:
```
git clone https://github.com/Stable-X/StableNormal.git
cd StableNormal
pip install -r requirements.txt
```
or directly build package:
```
pip install git+https://github.com/Stable-X/StableNormal.git
```

## Usage
To use the StableNormal pipeline, you can instantiate the model and apply it to an image as follows:

```python
import torch
from PIL import Image

# Load an image
input_image = Image.open("path/to/your/image.jpg")

# Create predictor instance
predictor = torch.hub.load("Stable-X/StableNormal", "StableNormal", trust_repo=True)

# Apply the model to the image
normal_image = predictor(input_image)

# Save or display the result
normal_image.save("output/normal_map.png")
```

**Additional Options:**

- If you need faster inference(10 times faster), use `StableNormal_turbo`:

```python
predictor = torch.hub.load("Stable-X/StableNormal", "StableNormal_turbo", trust_repo=True)
```

- If Hugging Face is not available from terminal, you could download the pretrained weights to `weights` dir:

```python
predictor = torch.hub.load("Stable-X/StableNormal", "StableNormal", trust_repo=True, local_cache_dir='./weights')
```
