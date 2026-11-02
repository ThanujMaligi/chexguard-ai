"""
Image Processing & Medical Radiograph Preprocessing Module for CheXGuard AI.
Handles standard chest radiograph normalization, aspect-ratio preservation,
CLAHE enhancement, and tensor conversion.
"""

from typing import Tuple, Union
import numpy as np
from PIL import Image, ImageOps
import torch
import torchvision.transforms as transforms


# CheXpert / ImageNet normalization parameters
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
DEFAULT_TARGET_SIZE = (224, 224)


def get_inference_transforms(target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE) -> transforms.Compose:
    """Returns the standardized PyTorch vision transforms for inference."""
    return transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])


def preprocess_radiograph(
    image_input: Union[str, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
    apply_clahe: bool = False
) -> Tuple[torch.Tensor, Image.Image]:
    """
    Preprocess an input radiograph into a PyTorch tensor and clean display image.
    
    Args:
        image_input: File path, PIL Image, or numpy array.
        target_size: Tuple (height, width) for model input.
        apply_clahe: Whether to apply contrast enhancement.
        
    Returns:
        tensor: Normalized PyTorch tensor of shape [1, 3, H, W]
        display_image: PIL RGB Image resized for UI visualization
    """
    if isinstance(image_input, str):
        pil_image = Image.open(image_input)
    elif isinstance(image_input, np.ndarray):
        pil_image = Image.fromarray(image_input)
    elif isinstance(image_input, Image.Image):
        pil_image = image_input.copy()
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # Ensure RGB mode
    pil_image = ImageOps.exif_transpose(pil_image)
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    display_image = pil_image.resize(target_size, Image.Resampling.BILINEAR)

    # Optional local contrast enhancement
    if apply_clahe:
        try:
            import cv2
            np_img = np.array(display_image)
            lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced = cv2.merge((cl, a, b))
            display_image = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB))
        except ImportError:
            pass

    # Transform to tensor
    transform = get_inference_transforms(target_size)
    tensor = transform(display_image).unsqueeze(0)  # Shape: [1, 3, H, W]

    return tensor, display_image


def denormalize_tensor(tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalize a normalized PyTorch tensor back to a uint8 RGB image for visualization.
    
    Args:
        tensor: PyTorch tensor [3, H, W] or [1, 3, H, W]
        
    Returns:
        np.ndarray: uint8 image of shape [H, W, 3] with values in [0, 255]
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    t = tensor.clone().detach().cpu()
    mean = torch.tensor(NORM_MEAN).view(3, 1, 1)
    std = torch.tensor(NORM_STD).view(3, 1, 1)
    t = t * std + mean
    t = torch.clamp(t, 0.0, 1.0)
    np_img = (t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return np_img
