"""
Procedural Radiograph & Clinical Pathology Sample Generator for CheXGuard AI.
Generates realistic thoracic radiographs demonstrating specific pathologies
(Normal, Pneumonia, Cardiomegaly, Pleural Effusion, Pneumothorax) and
an Out-of-Distribution test image for clinical safety verification.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "sample_cases")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_base_thoracic_radiograph(width: int = 512, height: int = 512) -> np.ndarray:
    """
    Synthesizes a realistic baseline human chest radiograph anatomy:
    Spine, mediastinum, bilateral lung fields, clavicles, ribs, and diaphragm.
    """
    img = np.zeros((height, width), dtype=np.float32)

    # 2D Coordinate Grid
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    cx, cy = width / 2.0, height / 2.0

    # 1. Mediastinum & Spine (Radiopaque center column - bright)
    spine_dist = np.abs(x - cx)
    spine_mask = np.exp(- (spine_dist ** 2) / (2.0 * (18.0 ** 2)))
    img += spine_mask * 140.0

    # 2. Thoracic Outer Soft Tissue Cage
    outer_dist = np.sqrt(((x - cx) / (width * 0.46)) ** 2 + ((y - cy * 1.05) / (height * 0.44)) ** 2)
    outer_cage = np.clip(1.0 - outer_dist, 0.0, 1.0)
    img += outer_cage * 80.0

    # 3. Bilateral Lung Fields (Radiolucent air - dark regions)
    # Left Hemithorax (anatomical right)
    lung_left_mask = (
        ((x - (cx - 100)) / 75.0) ** 2 +
        ((y - (cy - 10)) / 140.0) ** 2
    ) < 1.0
    # Right Hemithorax (anatomical left)
    lung_right_mask = (
        ((x - (cx + 100)) / 75.0) ** 2 +
        ((y - (cy - 10)) / 140.0) ** 2
    ) < 1.0

    img[lung_left_mask] -= 65.0
    img[lung_right_mask] -= 65.0

    # 4. Diaphragmatic Domes (Bright convex muscle at bottom of lungs)
    diaphragm_left = np.exp(- ((y - (cy + 120) + 0.003 * (x - (cx - 100)) ** 2) ** 2) / 300.0)
    diaphragm_right = np.exp(- ((y - (cy + 125) + 0.003 * (x - (cx + 100)) ** 2) ** 2) / 300.0)
    img += (diaphragm_left + diaphragm_right) * 110.0

    # Below diaphragm is dense abdomen (bright)
    img[y > (cy + 130)] += 70.0

    # 5. Normal Cardiac Silhouette (Heart shifted toward patient's left side: x > cx)
    heart_mask = (
        ((x - (cx + 35)) / 65.0) ** 2 +
        ((y - (cy + 60)) / 70.0) ** 2
    ) < 1.0
    img[heart_mask] += 75.0

    # 6. Clavicles (Bilateral horizontal collar bones)
    clav_left = np.exp(- ((y - (cy - 170) - 0.1 * (x - (cx - 110))) ** 2) / 30.0)
    clav_right = np.exp(- ((y - (cy - 170) + 0.1 * (x - (cx + 110))) ** 2) / 30.0)
    img += (clav_left + clav_right) * 90.0

    # 7. Subtle Rib Arches
    for r_offset in [-110, -70, -30, 10, 50, 90]:
        rib_left = np.exp(- ((y - (cy + r_offset) - 0.001 * (x - (cx - 100)) ** 2) ** 2) / 35.0)
        rib_right = np.exp(- ((y - (cy + r_offset) - 0.001 * (x - (cx + 100)) ** 2) ** 2) / 35.0)
        img += (rib_left + rib_right) * 28.0

    # Add realistic anatomical noise & texture
    np.random.seed(42)
    noise = np.random.normal(0, 4.5, (height, width))
    img += noise

    return np.clip(img, 15.0, 245.0)


def generate_case1_normal():
    """Case 1: Normal Chest Radiograph."""
    arr = create_base_thoracic_radiograph()
    img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    img = img.filter(ImageFilter.GaussianBlur(radius=0.7))
    img.save(os.path.join(OUTPUT_DIR, "case1_normal.png"))
    print("Saved case1_normal.png")


def generate_case2_pneumonia():
    """Case 2: Bacterial Pneumonia with Right Mid/Lower Lobe Consolidation."""
    arr = create_base_thoracic_radiograph()
    y, x = np.meshgrid(np.arange(512), np.arange(512), indexing='ij')
    # Infiltrate patch in right mid lung (patient's right: x < 256)
    consolidation = np.exp(- (((x - 165) / 45.0) ** 2 + ((y - 275) / 55.0) ** 2))
    arr += consolidation * 115.0
    # Additional patchy peribronchial opacities
    patch2 = np.exp(- (((x - 185) / 30.0) ** 2 + ((y - 320) / 40.0) ** 2))
    arr += patch2 * 85.0

    arr = np.clip(arr, 15.0, 245.0)
    img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    img = img.filter(ImageFilter.GaussianBlur(radius=0.7))
    img.save(os.path.join(OUTPUT_DIR, "case2_pneumonia.png"))
    print("Saved case2_pneumonia.png")


def generate_case3_cardiomegaly():
    """Case 3: Cardiomegaly with Grossly Enlarged Cardiac Silhouette."""
    arr = create_base_thoracic_radiograph()
    y, x = np.meshgrid(np.arange(512), np.arange(512), indexing='ij')
    # Markedly enlarged heart extending significantly into both hemithoraces
    enlarged_heart = (
        ((x - 290) / 115.0) ** 2 +
        ((y - 315) / 95.0) ** 2
    ) < 1.0
    arr[enlarged_heart] += 95.0
    # Ventricular apex extension
    apex = (
        ((x - 365) / 45.0) ** 2 +
        ((y - 355) / 45.0) ** 2
    ) < 1.0
    arr[apex] += 75.0

    arr = np.clip(arr, 15.0, 245.0)
    img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    img = img.filter(ImageFilter.GaussianBlur(radius=0.7))
    img.save(os.path.join(OUTPUT_DIR, "case3_cardiomegaly.png"))
    print("Saved case3_cardiomegaly.png")


def generate_case4_pleural_effusion():
    """Case 4: Pleural Effusion with Right Costophrenic Angle Blunting & Fluid Meniscus."""
    arr = create_base_thoracic_radiograph()
    y, x = np.meshgrid(np.arange(512), np.arange(512), indexing='ij')
    # Dense fluid collection obliterating right costophrenic sulcus (x < 210, y > 320)
    fluid_mask = (y > (320 + 0.003 * (x - 120) ** 2)) & (x < 225)
    arr[fluid_mask] += 125.0
    # Meniscus slope
    meniscus = np.exp(- ((y - (325 - 0.3 * (x - 100))) ** 2) / 120.0) * (x < 235)
    arr += meniscus * 60.0

    arr = np.clip(arr, 15.0, 245.0)
    img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    img = img.filter(ImageFilter.GaussianBlur(radius=0.7))
    img.save(os.path.join(OUTPUT_DIR, "case4_pleural_effusion.png"))
    print("Saved case4_pleural_effusion.png")


def generate_case5_pneumothorax():
    """Case 5: Left Apical Pneumothorax with Absent Peripheral Lung Markings."""
    arr = create_base_thoracic_radiograph()
    y, x = np.meshgrid(np.arange(512), np.arange(512), indexing='ij')
    # Hyperlucent peripheral apical zone (dark air space: x > 330, y < 250)
    ptx_zone = (x > 335) & (y < 260) & (y > 90)
    arr[ptx_zone] -= 60.0
    # Visceral pleural line (thin radiopaque curve where lung has retracted)
    pleural_line = np.exp(- ((x - (335 - 0.001 * (y - 180) ** 2)) ** 2) / 6.0) * (y < 265) * (y > 90)
    arr += pleural_line * 90.0

    arr = np.clip(arr, 15.0, 245.0)
    img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    img = img.filter(ImageFilter.GaussianBlur(radius=0.7))
    img.save(os.path.join(OUTPUT_DIR, "case5_pneumothorax.png"))
    print("Saved case5_pneumothorax.png")


def generate_case6_ood_invalid():
    """Case 6: Out-of-Distribution non-medical vivid photo (Nature/Sunset)."""
    # Create a colorful sunset landscape with trees and high RGB saturation
    img = Image.new("RGB", (512, 512))
    draw = ImageDraw.Draw(img)
    # Gradient sky
    for row in range(512):
        r = int(255 - row * 0.3)
        g = int(120 + row * 0.2)
        b = int(40 + row * 0.4)
        draw.line([(0, row), (512, row)], fill=(min(255, r), min(255, g), min(255, b)))
    # Bright sun
    draw.ellipse([210, 140, 300, 230], fill=(255, 240, 90))
    # Green hills
    draw.polygon([(0, 400), (200, 350), (380, 420), (512, 380), (512, 512), (0, 512)], fill=(34, 139, 34))
    img.save(os.path.join(OUTPUT_DIR, "case6_ood_invalid.png"))
    print("Saved case6_ood_invalid.png")


if __name__ == "__main__":
    generate_case1_normal()
    generate_case2_pneumonia()
    generate_case3_cardiomegaly()
    generate_case4_pleural_effusion()
    generate_case5_pneumothorax()
    generate_case6_ood_invalid()
    print("All sample cases successfully synthesized!")
