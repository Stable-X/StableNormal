# StableNormal: Reducing Diffusion Variance for Stable and Sharp Normal

This repository represents the official implementation of the paper titled "Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation".

[![Website](https://raw.githubusercontent.com/prs-eth/Marigold/main/doc/badges/badge-website.svg)](https://stable-x.github.io/StableNormal)
[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2406.16864)
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face%20-Space-yellow)](https://huggingface.co/spaces/Stable-X/StableNormal)
[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face%20-Model-green)](https://huggingface.co/Stable-X/stable-normal-v0-1)
[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)

We propose StableNormal, which tailors the diffusion priors for monocular normal estimation. Unlike prior diffusion-based works, we focus on enhancing estimation stability by reducing the inherent stochasticity of diffusion models ( i.e. , Stable Diffusion). This enables “Stable-and-Sharp” normal estimation, which outperforms multiple baselines (try [Compare](https://huggingface.co/spaces/Stable-X/normal-estimation-arena)), and improves various real-world applications (try [Demo](https://huggingface.co/spaces/Stable-X/StableNormal)). 

![teaser](doc/StableNormal-Teaser.jpg)

