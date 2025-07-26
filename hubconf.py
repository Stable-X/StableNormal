from __future__ import annotations

dependencies = ["torch", "numpy", "diffusers", "PIL", "transformers"]

import enum
import os
from typing import Optional, Tuple, Union
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation, AutoModelForImageSegmentation

class DataType(enum.Enum):
    INDOOR = "indoor"  # No masking
    OBJECT = "object"  # Mask background using BiRefNet or alpha channel
    OUTDOOR = "outdoor"  # Mask vegetation and sky using Mask2Former

class SegmentationHandler:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.mask2former_processor = None
        self.mask2former_model = None
        self.birefnet_model = None
        
    def _lazy_load_mask2former(self):
        """Lazy loading of the Mask2Former model"""
        if self.mask2former_model is None:
            self.mask2former_processor = AutoImageProcessor.from_pretrained(
                "facebook/mask2former-swin-large-cityscapes-semantic"
            )
            self.mask2former_model = Mask2FormerForUniversalSegmentation.from_pretrained(
                "facebook/mask2former-swin-large-cityscapes-semantic"
            ).to(self.device)
            self.mask2former_model.eval()

    def _lazy_load_birefnet(self):
        """Lazy loading of the BiRefNet model"""
        if self.birefnet_model is None:
            self.birefnet_model = AutoModelForImageSegmentation.from_pretrained(
                'zhengpeng7/BiRefNet',
                trust_remote_code=True
            ).to(self.device)
            self.birefnet_model.eval()

    def _get_birefnet_mask(self, image: Image.Image) -> np.ndarray:
        """Get object mask using BiRefNet"""
        # Data settings
        image_size = (1024, 1024)
        transform_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        input_images = transform_image(image).unsqueeze(0).to(self.device)

        # Prediction
        with torch.no_grad():
            preds = self.birefnet_model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image.size)
        mask_np = np.array(mask)
        
        return (mask_np > 128).astype(np.uint8)

    def _get_mask2former_mask(self, image: Image.Image) -> np.ndarray:
        """Get outdoor mask using Mask2Former"""
        inputs = self.mask2former_processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.mask2former_model(**inputs)
            
        predicted_semantic_map = self.mask2former_processor.post_process_semantic_segmentation(
            outputs, 
            target_sizes=[image.size[::-1]]
        )[0].cpu().numpy()
        
        # Mask vegetation (class 9) and sky (class 10)
        mask = ~np.isin(predicted_semantic_map, [9, 10])
        return mask.astype(np.uint8)

    def get_mask(self, image: Image.Image, data_type: DataType) -> Optional[np.ndarray]:
        """
        Get segmentation mask based on data type.
        
        Args:
            image: Input PIL Image
            data_type: Type of data processing required
            
        Returns:
            Optional numpy array mask where 1 indicates areas to keep
        """
        if data_type == DataType.INDOOR:
            return None
            
        if data_type == DataType.OBJECT:
            self._lazy_load_birefnet()
            return self._get_birefnet_mask(image)
        else:  # OUTDOOR
            self._lazy_load_mask2former()
            return self._get_mask2former_mask(image)

class Predictor:
    def __init__(self, model, yoso_version: Optional[str] = None):
        self.model = model
        self.segmentation_handler = SegmentationHandler()
        self.yoso_version = yoso_version
    
    def to(self, device: str = "cuda", dtype: torch.dtype = torch.float16):
        self.model.to(device, dtype)
        self.segmentation_handler.device = device
        return self
    
    def _apply_mask(self, 
                    prediction: np.ndarray, 
                    mask: Optional[np.ndarray]
                   ) -> np.ndarray:
        """Apply mask to normal map prediction if mask exists"""
        if mask is not None:
            prediction = prediction.copy()
            prediction[mask == 0] = 1
        return prediction
    
    def _process_rgba_image(self, img: Image.Image) -> Tuple[Image.Image, np.ndarray]:
        """
        Process RGBA image by extracting alpha channel as mask and creating white background
        
        Args:
            img: RGBA PIL Image
            
        Returns:
            Tuple of (RGB image with white background, alpha mask)
        """
        # Split alpha channel
        rgb = img.convert('RGB')
        alpha = img.split()[-1]
        
        # Create white background image
        white_bg = Image.new('RGB', img.size, (255, 255, 255))
        
        # Composite the image onto white background
        composite = Image.composite(rgb, white_bg, alpha)
        
        # Convert alpha to numpy mask
        alpha_mask = (np.array(alpha) > 128).astype(np.uint8)
        
        return composite, alpha_mask
    
    @torch.no_grad()
    def __call__(
        self, 
        img: Image.Image, 
        resolution: int = 1024,
        match_input_resolution: bool = True,
        data_type: Union[DataType, str] = DataType.INDOOR,
        num_inference_steps: int = None
    ) -> Image.Image:
        """
        Generate normal map from input image.
        
        Args:
            img: Input PIL Image
            resolution: Target processing resolution
            match_input_resolution: Whether to match input image resolution
            data_type: Type of data (indoor/object/outdoor) affecting masking
            num_inference_steps: Optional number of inference steps
            
        Returns:
            PIL Image containing the normal map
        """
        if isinstance(data_type, str):
            data_type = DataType(data_type.lower())
        
        if self.yoso_version:
            version_str = self.yoso_version.split('-')[-1] 
            version_num = float(version_str[1:].replace('-', '.')) 
            if version_num > 1.5 and data_type != DataType.OBJECT:
                import warnings
                warnings.warn(
                    f"Current version ({self.yoso_version}) is not optimized for scene normal estimation. "
                    "For better results with indoor/outdoor scenes, please use version v1.5 or earlier.",
                    UserWarning
                )
        
        # Handle RGBA images
        alpha_mask = None
        orig_size = img.size
        if img.mode == 'RGBA':
            img, alpha_mask = self._process_rgba_image(img)
            img = resize_image(img, resolution)
            alpha_mask = Image.fromarray(alpha_mask).resize(img.size, Image.Resampling.NEAREST)
            alpha_mask = np.array(alpha_mask)
            mask = alpha_mask
        else:
            # Regular RGB image processing
            img = resize_image(img, resolution)
            mask = self.segmentation_handler.get_mask(img, data_type) if data_type != DataType.INDOOR else None
        
        # Generate normal map
        kwargs = {}
        if num_inference_steps is not None:
            kwargs['num_inference_steps'] = num_inference_steps
            
        pipe_out = self.model(
            img,
            match_input_resolution=match_input_resolution,
            **kwargs
        )

        # Apply mask if exists
        prediction = pipe_out.prediction[0]
        prediction = self._apply_mask(prediction, mask)
        
        # Convert prediction to image
        normal_map = (prediction.clip(-1, 1) + 1) / 2
        normal_map = (normal_map * 255).astype(np.uint8)
        normal_map = Image.fromarray(normal_map)
        
        # Resize back to original dimensions if needed
        if match_input_resolution:
            normal_map = normal_map.resize(
                orig_size, 
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

def StableNormal(local_cache_dir: Optional[str] = None, device="cuda:0", 
                 yoso_version='yoso-normal-v1-5', diffusion_version='stable-normal-v0-1') -> Predictor:
    """Load the StableNormal pipeline and return a Predictor instance."""
    
    version_str = yoso_version.split('-')[-1] 
    version_num = float(version_str[1:].replace('-', '.'))  
    
    if version_num < 1.5:
        from stablenormal.pipeline_yoso_normal import YOSONormalsPipeline
        from stablenormal.pipeline_stablenormal import StableNormalPipeline
        from stablenormal.scheduler.heuristics_ddimsampler import HEURI_DDIMScheduler
        use_safety_checker = None
    else:
        from nirne.pipeline_yoso_normal import YOSONormalsPipeline
        from nirne.pipeline_stablenormal import StableNormalPipeline
        from nirne.scheduler.heuristics_ddimsampler import HEURI_DDIMScheduler
        use_safety_checker = True
    
    yoso_weight_path = os.path.join(local_cache_dir if local_cache_dir else "Stable-X", yoso_version)
    diffusion_weight_path = os.path.join(local_cache_dir if local_cache_dir else "Stable-X", diffusion_version)
    
    common_kwargs = {
        "variant": "fp16",
        "torch_dtype": torch.float16,
        "trust_remote_code": True
    }
    
    if version_num < 1.5:
        common_kwargs["safety_checker"] = None
    
    x_start_pipeline = YOSONormalsPipeline.from_pretrained(
        yoso_weight_path, **common_kwargs).to(device)
    
    pipe = StableNormalPipeline.from_pretrained(
        diffusion_weight_path,
        **common_kwargs,
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
    
    return Predictor(pipe, yoso_version=yoso_version)

def StableNormal_turbo(local_cache_dir: Optional[str] = None, device="cuda:0", 
                      yoso_version='yoso-normal-v1-5') -> Predictor:
    """Load the StableNormal_turbo pipeline for a faster inference."""
    
    version_str = yoso_version.split('-')[-1] 
    version_num = float(version_str[1:].replace('-', '.')) 
    
    if version_num < 1.5:
        from stablenormal.pipeline_yoso_normal import YOSONormalsPipeline
    else:
        from nirne.pipeline_yoso_normal import YOSONormalsPipeline
    
    yoso_weight_path = os.path.join(local_cache_dir if local_cache_dir else "Stable-X", yoso_version)
    
    kwargs = {
        "trust_remote_code": True,
        "variant": "fp16",
        "torch_dtype": torch.float16,
        "t_start": 0
    }
    
    if version_num < 1.5:
        kwargs["safety_checker"] = None
    
    pipe = YOSONormalsPipeline.from_pretrained(yoso_weight_path, **kwargs).to(device)
    
    return Predictor(pipe, yoso_version=yoso_version)

def resize_image(input_image: Image.Image, resolution: int = 1024) -> Image.Image:
    """
    Resize image to target resolution while maintaining aspect ratio and ensuring dimensions are multiples of 64.
    
    Args:
        input_image: PIL Image to resize
        resolution: Target resolution for the shorter dimension
        
    Returns:
        Resized PIL Image
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
    return resized_image