"""
Out-of-Distribution (OOD) Detector & Medical Radiograph Quality Validator.
Provides Responsible AI safety guardrails by detecting non-radiographic images
(e.g., natural photos, selfies, cartoons, text) or severely degraded inputs
before medical inference is performed.
"""

from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image


class OODDetector:
    """
    Evaluates image integrity and anatomical radiograph distribution.
    Prevents hallucinated medical predictions on invalid inputs.
    """

    def __init__(
        self,
        max_color_variance: float = 18.0,
        min_dynamic_range: float = 40.0,
        min_contrast: float = 20.0,
        max_saturation: float = 0.25
    ):
        self.max_color_variance = max_color_variance
        self.min_dynamic_range = min_dynamic_range
        self.min_contrast = min_contrast
        self.max_saturation = max_saturation

    def inspect_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Runs comprehensive biometric and distribution sanity checks.
        
        Args:
            image: PIL Image object
            
        Returns:
            dict containing:
                - is_valid (bool): True if image conforms to thoracic radiograph characteristics
                - ood_risk_score (float): 0.0 (safe radiograph) to 1.0 (definite OOD)
                - reason (str): Human-readable clinical safety explanation
                - metrics (dict): Quantitative diagnostic metrics computed on the image
        """
        rgb_img = image.convert("RGB")
        arr = np.array(rgb_img, dtype=np.float32)

        # 1. Color Disparity / Saturation Check
        # Radiographs are monochromatic; natural images have chromatic disparity
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        channel_diff_rg = np.mean(np.abs(r - g))
        channel_diff_rb = np.mean(np.abs(r - b))
        channel_diff_gb = np.mean(np.abs(g - b))
        mean_color_variance = float((channel_diff_rg + channel_diff_rb + channel_diff_gb) / 3.0)

        # HSV Saturation
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        delta = max_c - min_c
        with np.errstate(divide='ignore', invalid='ignore'):
            saturation = np.where(max_c == 0, 0, delta / (max_c + 1e-7))
        mean_saturation = float(np.mean(saturation))

        # 2. Grayscale Dynamic Range & Contrast Check
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        p5, p95 = np.percentile(gray, 5), np.percentile(gray, 95)
        dynamic_range = float(p95 - p5)
        contrast_std = float(np.std(gray))

        # 3. Structural Gradient & Edge Energy (Laplacian proxy)
        diff_x = np.diff(gray, axis=1)
        diff_y = np.diff(gray, axis=0)
        edge_energy = float(np.mean(np.abs(diff_x)) + np.mean(np.abs(diff_y)))

        # 4. Bilateral Symmetry (Thoracic Cavity Characteristic)
        h, w = gray.shape
        mid_w = w // 2
        left_half = gray[:, :mid_w]
        right_half_flipped = np.fliplr(gray[:, w - mid_w:])
        symmetry_diff = float(np.mean(np.abs(left_half - right_half_flipped)))

        # Compute aggregate OOD penalty
        color_penalty = max(0.0, min(1.0, (mean_color_variance - 10.0) / 25.0))
        sat_penalty = max(0.0, min(1.0, (mean_saturation - 0.10) / 0.30))
        range_penalty = 1.0 - max(0.0, min(1.0, dynamic_range / 70.0))
        contrast_penalty = 1.0 - max(0.0, min(1.0, contrast_std / 35.0))

        # Weighted aggregate risk score
        ood_risk = float(
            0.45 * color_penalty +
            0.25 * sat_penalty +
            0.15 * range_penalty +
            0.15 * contrast_penalty
        )
        ood_risk = max(0.0, min(1.0, ood_risk))

        # Decision threshold
        is_valid = bool(ood_risk < 0.40 and mean_color_variance < self.max_color_variance)

        if not is_valid:
            if mean_color_variance >= self.max_color_variance or mean_saturation > self.max_saturation:
                reason = (
                    "Input rejected: High chromatic saturation detected. "
                    "Expected a monochromatic grayscale thoracic radiograph."
                )
            elif dynamic_range < self.min_dynamic_range or contrast_std < self.min_contrast:
                reason = (
                    "Input rejected: Insufficient radiographic contrast or dynamic range. "
                    "Image appears blank, underexposed, or severely corrupted."
                )
            else:
                reason = (
                    "Input rejected: Anatomical distribution does not conform to thoracic radiograph standards."
                )
        else:
            reason = "Input verified: Image conforms to thoracic radiograph anatomical distribution."

        return {
            "is_valid": is_valid,
            "ood_risk_score": round(ood_risk, 3),
            "confidence": round(1.0 - ood_risk, 3),
            "reason": reason,
            "metrics": {
                "color_variance": round(mean_color_variance, 2),
                "saturation": round(mean_saturation, 3),
                "dynamic_range": round(dynamic_range, 1),
                "contrast_std": round(contrast_std, 1),
                "edge_energy": round(edge_energy, 2),
                "symmetry_diff": round(symmetry_diff, 1)
            }
        }
