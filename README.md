# **StableNormal: Reducing Diffusion Variance for Stable and Sharp Normal**<br>

[Chongjie Ye*](https://github.com/hugoycj), [Lingteng Qiu*](https://lingtengqiu.github.io/), [Xiaodong Gu](https://github.com/gxd1994), [Qi Zuo](https://github.com/hitsz-zuoqi), [Yushuang Wu](https://scholar.google.com/citations?hl=zh-TW&user=x5gpN0sAAAAJ), [Zilong Dong](https://scholar.google.com/citations?user=GHOQKCwAAAAJ), [Liefeng Bo](https://research.cs.washington.edu/istc/lfb/), [Yuliang Xiu#](https://xiuyuliang.cn/), [Xiaoguang Han#](https://gaplab.cuhk.edu.cn/)<br>

\* Equal contribution <br>
\# Corresponding Author


<h3 align="center">SIGGRAPH Asia 2024 (Journal Track)</h3>

<div align="center">


[![Website](https://raw.githubusercontent.com/prs-eth/Marigold/main/doc/badges/badge-website.svg)](https://stable-x.github.io/StableNormal) 
[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2406.16864) 
[![ModelScope](https://img.shields.io/badge/%20ModelScope%20-Space-blue)](https://modelscope.cn/studios/Damo_XR_Lab/StableNormal) 
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face%20-Space-yellow)](https://huggingface.co/spaces/Stable-X/StableNormal) 
[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face%20-Model-green)](https://huggingface.co/Stable-X/stable-normal-v0-1) 
[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0) 

</div>


We propose StableNormal, which tailors the diffusion priors for monocular normal estimation. Unlike prior diffusion-based works, we focus on enhancing estimation stability by reducing the inherent stochasticity of diffusion models ( i.e. , Stable Diffusion). This enables “Stable-and-Sharp” normal estimation, which outperforms multiple baselines (try [Compare](https://huggingface.co/spaces/Stable-X/normal-estimation-arena)), and improves various real-world applications (try [Demo](https://huggingface.co/spaces/Stable-X/StableNormal)). 

![teaser](doc/StableNormal-Teaser.jpg)

## News
- StableNormal **YOSO** is now avaliable on [ModelScope]( https://modelscope.cn/studios/Damo_XR_Lab/StableNormal ) . We invite you to explore its features!  :fire::fire::fire: (10.11, 2024 UTC)
- StableNormal is got accepted by SIGGRAPH Asia 2024. (**Journal Track)**) (09.11, 2024 UTC)
- Release StableDelight :fire::fire::fire: (09.07, 2024 UTC)
- Release StableNormal :fire::fire::fire: (08.27, 2024 UTC)

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



**Compute Metrics:**

This section provides guidance on evaluating your normal predictor using the DIODE dataset.

**Step 1**: Prepare Your Results Folder

First, make sure you have generated a normal map and structured your results folder as shown below:


```bash
├── YOUR-FOLDER-NAME 
│   ├── scan_00183_00019_00183_indoors_000_010_gt.png
│   ├── scan_00183_00019_00183_indoors_000_010_init.png
│   ├── scan_00183_00019_00183_indoors_000_010_ref.png
│   ├── scan_00183_00019_00183_indoors_000_010_step0.png
│   ├── scan_00183_00019_00183_indoors_000_010_step1.png
│   ├── scan_00183_00019_00183_indoors_000_010_step2.png
│   ├── scan_00183_00019_00183_indoors_000_010_step3.png
```


**Step 2**: Compute Metric Values

Once your results folder is set up, you can compute the metrics for your normal predictions by running the following scripts:

```bash
# compute metrics
python ./stablenormal/metrics/compute_metric.py -i ${YOUR-FOLDER-NAME}

# compute variance
python ./stablenormal/metrics/compute_variance.py -i ${YOUR-FOLDER-NAME}
```

Replace ${YOUR-FOLDER-NAME}; with the actual name of your results folder. Following these steps will allow you to effectively evaluate your normal predictor's performance on the DIODE dataset.

## Citation

```bibtex
@article{ye2024stablenormal,
  title={StableNormal: Reducing Diffusion Variance for Stable and Sharp Normal},
  author={Ye, Chongjie and Qiu, Lingteng and Gu, Xiaodong and Zuo, Qi and Wu, Yushuang and Dong, Zilong and Bo, Liefeng and Xiu, Yuliang and Han, Xiaoguang},
  journal={ACM Transactions on Graphics (TOG)},
  year={2024},
  publisher={ACM New York, NY, USA}
}
```
