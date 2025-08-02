from __future__ import annotations

import functools
import os
import sys
import glob
import shutil
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from stablenormal.pipeline_yoso_normal import YOSONormalsPipeline
from stablenormal.pipeline_stablenormal import StableNormalPipeline
from stablenormal.scheduler.heuristics_ddimsampler import HEURI_DDIMScheduler

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def resize_image(input_image, resolution=1024):
    # Ensure input_image is a PIL Image object
    if not isinstance(input_image, Image.Image):
        raise ValueError("input_image should be a PIL Image object")

    # Convert image to numpy array
    input_image_np = np.asarray(input_image)

    # Get image dimensions
    H, W, C = input_image_np.shape
    H = float(H)
    W = float(W)
    
    # Calculate the scaling factor
    k = float(resolution) / max(H, W)
    
    # Determine new dimensions
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    
    # Resize the image using PIL's resize method
    img = input_image.resize((W, H), Image.Resampling.LANCZOS)
    
    return img

def process_image(pipe, image_path):
    name_base = os.path.splitext(os.path.basename(image_path))[0]

    # Load and preprocess input image
    input_image = Image.open(image_path)
    input_image = resize_image(input_image)
    
    # Generate normal map
    pipe_out = pipe(
        input_image,
        match_input_resolution=False,
        processing_resolution=max(input_image.size),
        num_inference_steps=2
    )

    # Visualize and save normal map
    normal_colored = pipe.image_processor.visualize_normals(pipe_out.prediction)
    out_path = f"{name_base}.png"
    normal_colored[-1].save(out_path)
    
    return out_path

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_directory>")
        sys.exit(1)

    # Initialize models
    device = DEFAULT_DEVICE
    
    print("Loading normal estimation model...")
    x_start_pipeline = YOSONormalsPipeline.from_pretrained(
        'weights/yoso-normal-v1-4', trust_remote_code=True, variant="fp16", torch_dtype=torch.float16, t_start=0).to(device)
    pipe = StableNormalPipeline.from_pretrained('weights/stable-normal-v0-1', trust_remote_code=True,
                                                variant="fp16", torch_dtype=torch.float16,
                                                scheduler=HEURI_DDIMScheduler(prediction_type='sample', 
                                                                              beta_start=0.00085, beta_end=0.0120, 
                                                                              beta_schedule = "scaled_linear"))
    pipe.x_start_pipeline = x_start_pipeline
    pipe.to(device)
    pipe.prior.to(device, torch.float16)
    
    try:
        import xformers
        pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        print("XFormers not available, running without memory optimizations")
    
    # Setup input/output directories
    input_dir = sys.argv[1]
    output_dir = os.path.join(input_dir, 'normals')
    os.makedirs(output_dir, exist_ok=True)

    # Process all images
    image_patterns = [
        os.path.join(input_dir, "images", "*.jpg"),
        os.path.join(input_dir, "images", "*.JPG"),
        os.path.join(input_dir, "images", "*.png")
    ]
    
    image_paths = []
    for pattern in image_patterns:
        image_paths.extend(glob.glob(pattern))
    
    print(f"Found {len(image_paths)} images to process")
    
    for image_path in tqdm(image_paths, desc="Processing images"):
        out_path = process_image(pipe, image_path)
        final_path = os.path.join(output_dir, os.path.basename(out_path))
        shutil.move(out_path, final_path)

if __name__ == "__main__":
    main()