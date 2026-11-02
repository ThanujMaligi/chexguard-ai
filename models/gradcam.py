"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Module for CheXGuard AI.
Generates localized visual heatmaps highlighting anatomical thoracic regions
driving the model's pathology predictions.
"""

from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.cm as cm


class GradCAM:
    """
    Computes visual explanations for decisions of CheXGuardNet convolutional layers.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        """Registers forward and backward hooks on the target convolutional layer."""
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self._hook_handles.extend([h1, h2])

    def remove_hooks(self):
        """Removes registered hooks to prevent memory leaks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        class_idx: int
    ) -> np.ndarray:
        """
        Computes 2D Grad-CAM heatmap for a specific class index.
        
        Args:
            input_tensor: Tensor of shape [1, 3, H, W]
            class_idx: Target pathology class index
            
        Returns:
            heatmap: 2D numpy array [H, W] normalized to [0, 1]
        """
        self.model.eval()
        self.model.zero_grad()

        input_var = input_tensor.clone().requires_grad_(True)
        logits = self.model(input_var)

        target_score = logits[0, class_idx]
        target_score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to capture gradients or activations in Grad-CAM.")

        # Spatial Global Average Pooling of gradients
        # Shape of gradients: [1, C, h, w]
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # Linear combination of weighted activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # [1, 1, h, w]
        cam = F.relu(cam)  # Only features with positive influence

        # Resize to input tensor spatial dimensions
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)

        cam_np = cam.squeeze().detach().cpu().numpy()

        # Min-max normalization
        cam_min, cam_max = np.min(cam_np), np.max(cam_np)
        if cam_max - cam_min > 1e-8:
            cam_norm = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam_np)

        return cam_norm

    def overlay_heatmap(
        self,
        original_image: Image.Image,
        heatmap: np.ndarray,
        colormap_name: str = "jet",
        alpha: float = 0.45
    ) -> Image.Image:
        """
        Overlays the 2D Grad-CAM heatmap onto the original radiograph.
        
        Args:
            original_image: PIL RGB image
            heatmap: 2D numpy array [H, W] in [0, 1]
            colormap_name: Matplotlib colormap (e.g. 'jet', 'inferno', 'turbo')
            alpha: Heatmap blend opacity factor
            
        Returns:
            blended_image: PIL Image with overlaid visual heatmap
        """
        w, h = original_image.size
        # Resize heatmap to match image dimensions if different
        if heatmap.shape != (h, w):
            heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
            heatmap_resized = np.array(heatmap_pil.resize((w, h), Image.Resampling.BILINEAR)) / 255.0
        else:
            heatmap_resized = heatmap

        # Apply colormap
        import matplotlib.pyplot as plt
        try:
            import matplotlib as mpl
            cmap = mpl.colormaps[colormap_name]
        except Exception:
            cmap = cm.get_cmap(colormap_name)
        colored_cam = cmap(heatmap_resized)[:, :, :3]  # Strip alpha channel -> [H, W, 3] in [0, 1]
        colored_cam_uint8 = (colored_cam * 255).astype(np.uint8)

        # Blend with original
        orig_np = np.array(original_image.convert("RGB"))
        blended = (1.0 - alpha) * orig_np + alpha * colored_cam_uint8
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        return Image.fromarray(blended)
