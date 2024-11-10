from __future__ import annotations

import os
from typing import Optional, Tuple
import torch
import numpy as np
from PIL import Image

dependencies = ["torch", "numpy", "diffusers", "PIL"]

from stablenormal.pipeline_yoso_normal import YOSONormalsPipeline
from stablenormal.pipeline_stablenormal import StableNormalPipeline
from stablenormal.scheduler.heuristics_ddimsampler import HEURI_DDIMScheduler

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def resize_image(input_image: Image.Image, resolution: int = 1024) -> Tuple[Image.Image, Tuple[int, int], Tuple[float, float]]:
    """
    Resize image to target resolution while maintaining aspect ratio and ensuring dimensions are multiples of 64.
    
    Args:
        input_image: PIL Image to resize
        resolution: Target resolution for the shorter dimension
        
    Returns:
        Tuple containing:
        - Resized PIL Image
        - Original dimensions (height, width)
        - Scaling factors (height_scale, width_scale)
    """
    if not isinstance(input_image, Image.Image):
        raise ValueError("input_image should be a PIL Image object")

    input_image_np = np.asarray(input_image)
    H, W, C = input_image_np.shape
    H, W = float(H), float(W)
    
    k = float(resolution) / max(H, W)
    new_H = H * k
    new_W = W * k
    new_H = int(np.round(new_H / 64.0)) * 64
    new_W = int(np.round(new_W / 64.0)) * 64
    
    resized_image = input_image.resize((new_W, new_H), Image.Resampling.LANCZOS)
    return resized_image, (int(H), int(W)), (new_H / H, new_W / W)

class Predictor:
    def __init__(self, model):
        self.model = model
        try:
            import xformers
            self.model.enable_xformers_memory_efficient_attention()
        except ImportError:
            pass

    def to(self, device: str = DEFAULT_DEVICE, dtype: torch.dtype = torch.float16):
        self.model.to(device, dtype)
        return self
    
    @torch.no_grad()
    def __call__(
        self, 
        img: Image.Image, 
        resolution: int = 1024,
        match_input_resolution: bool = True
    ) -> Image.Image:
        """
        Generate normal map from input image.
        
        Args:
            img: Input PIL Image
            resolution: Target processing resolution
            match_input_resolution: Whether to match input image resolution
            
        Returns:
            PIL Image containing the normal map
        """
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        # Resize image
        img, original_dims, scaling_factors = resize_image(img, resolution)

        # Generate normal map
        pipe_out = self.model(
            img,
            match_input_resolution=match_input_resolution,
            processing_resolution=max(img.size)
        )
        
        # Convert prediction to image
        normal_map = (pipe_out.prediction.clip(-1, 1) + 1) / 2
        normal_map = (normal_map[0] * 255).astype(np.uint8)
        normal_map = Image.fromarray(normal_map)
        
        # Resize back to original dimensions if needed
        if match_input_resolution:
            normal_map = normal_map.resize(
                (original_dims[1], original_dims[0]), 
                Image.Resampling.LANCZOS
            )
            
        return normal_map
    
    def visualize_normals(self, img: Image.Image, **kwargs) -> Image.Image:
        """Convert normal map to RGB visualization."""
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        prediction = np.array(img).astype(np.float32) / 255.0 * 2 - 1
        prediction = np.expand_dims(prediction, axis=0)
        return self.model.image_processor.visualize_normals(prediction)[-1]

def StableNormal(
    local_cache_dir: Optional[str] = None,
    device: str = "cuda:0",
    yoso_version: str = 'yoso-normal-v1-4',
    diffusion_version: str = 'stable-normal-v0-1'
) -> Predictor:
    """
    Load the full StableNormal pipeline.
    
    Args:
        local_cache_dir: Path to model weights directory
        device: Device to load model on
        yoso_version: Version of YOSO model to use
        diffusion_version: Version of diffusion model to use
        
    Returns:
        Predictor instance
    """
    yoso_weight_path = os.path.join(local_cache_dir if local_cache_dir else "weights", yoso_version)
    diffusion_weight_path = os.path.join(local_cache_dir if local_cache_dir else "weights", diffusion_version)
    
    x_start_pipeline = YOSONormalsPipeline.from_pretrained(
        yoso_weight_path,
        trust_remote_code=True,
        safety_checker=None,
        variant="fp16",
        torch_dtype=torch.float16
    ).to(device)
    
    # Load diffusion pipeline
    pipe = StableNormalPipeline.from_pretrained(
        diffusion_weight_path,
        trust_remote_code=True,
        safety_checker=None,
        variant="fp16",
        torch_dtype=torch.float16,
        scheduler=HEURI_DDIMScheduler(
            prediction_type='sample',
            beta_start=0.00085,
            beta_end=0.0120,
            beta_schedule="scaled_linear"
        )
    )

    pipe.x_start_pipeline = x_start_pipeline
    pipe.to(device)
    pipe.prior.to(device, torch.float16)
    
    return Predictor(pipe)

def StableNormal_turbo(
    local_cache_dir: Optional[str] = None,
    device: str = "cuda:0",
    yoso_version: str = 'yoso-normal-v1-4'
) -> Predictor:
    """
    Load the faster StableNormal_turbo pipeline.
    
    Args:
        local_cache_dir: Path to model weights directory
        device: Device to load model on
        yoso_version: Version of YOSO model to use
        
    Returns:
        Predictor instance
    """
    yoso_weight_path = os.path.join(local_cache_dir if local_cache_dir else "weights", yoso_version)
    pipe = YOSONormalsPipeline.from_pretrained(
        yoso_weight_path,
        trust_remote_code=True,
        safety_checker=None,
        variant="fp16",
        torch_dtype=torch.float16,
        t_start=0
    ).to(device)

    return Predictor(pipe)

def _test_run():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, required=True, help="Input image file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output image file")
    parser.add_argument("--mode", type=str, default="StableNormal_turbo", help="Mode of operation")

    args = parser.parse_args()

    predictor_func = StableNormal_turbo if args.mode == "StableNormal_turbo" else StableNormal
    predictor = predictor_func(local_cache_dir='./weights', device="cuda:0")
    
    image = Image.open(args.input)
    with torch.inference_mode():
        normal_image = predictor(image) 
    normal_image.save(args.output)

if __name__ == "__main__":
    _test_run()
