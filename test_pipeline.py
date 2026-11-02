"""
Automated Comprehensive Test Suite for CheXGuard AI.
Verifies input preprocessing, OOD detection, model inference, Grad-CAM generation,
Monte Carlo uncertainty estimation, and PDF clinical report generation.
"""

import os
import unittest
import numpy as np
import torch
from PIL import Image

from utils.image_processing import preprocess_radiograph, denormalize_tensor
from utils.report_generator import compute_triage_priority, generate_markdown_report, generate_pdf_report
from models.backbone import CheXGuardNet, build_calibrated_model, TARGET_PATHOLOGIES, URGENCY_WEIGHTS
from models.gradcam import GradCAM
from models.uncertainty import MonteCarloUncertaintyEngine
from models.ood_detector import OODDetector


SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "data", "sample_cases")


class TestCheXGuardPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = build_calibrated_model()
        cls.ood_detector = OODDetector()
        cls.uncertainty_engine = MonteCarloUncertaintyEngine(cls.model, TARGET_PATHOLOGIES)
        cls.gradcam = GradCAM(cls.model, cls.model.get_target_layer())

    @classmethod
    def tearDownClass(cls):
        cls.gradcam.remove_hooks()

    def test_01_sample_cases_exist(self):
        """Verifies that all 6 required sample cases are present on disk."""
        expected_files = [
            "case1_normal.png",
            "case2_pneumonia.png",
            "case3_cardiomegaly.png",
            "case4_pleural_effusion.png",
            "case5_pneumothorax.png",
            "case6_ood_invalid.png"
        ]
        for fname in expected_files:
            fpath = os.path.join(SAMPLE_DIR, fname)
            self.assertTrue(os.path.exists(fpath), f"Sample case missing: {fname}")

    def test_02_image_preprocessing(self):
        """Tests image loading, normalization, and tensor shape."""
        normal_path = os.path.join(SAMPLE_DIR, "case1_normal.png")
        tensor, display_img = preprocess_radiograph(normal_path)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        self.assertEqual(display_img.size, (224, 224))

        # Check denormalization
        denorm = denormalize_tensor(tensor)
        self.assertEqual(denorm.shape, (224, 224, 3))
        self.assertEqual(denorm.dtype, np.uint8)

    def test_03_ood_detector(self):
        """Tests that medical radiograph passes and non-medical colorful photo is rejected."""
        normal_path = os.path.join(SAMPLE_DIR, "case1_normal.png")
        ood_path = os.path.join(SAMPLE_DIR, "case6_ood_invalid.png")

        normal_img = Image.open(normal_path)
        ood_img = Image.open(ood_path)

        res_normal = self.ood_detector.inspect_image(normal_img)
        res_ood = self.ood_detector.inspect_image(ood_img)

        # Radiograph must pass
        self.assertTrue(res_normal["is_valid"], "Valid radiograph was rejected by OOD detector!")
        self.assertLess(res_normal["ood_risk_score"], 0.40)

        # Non-medical photo must be rejected
        self.assertFalse(res_ood["is_valid"], "OOD non-medical image was falsely accepted!")
        self.assertGreaterEqual(res_ood["ood_risk_score"], 0.40)

    def test_04_model_forward_pass(self):
        """Tests CheXGuardNet forward pass and probability output."""
        dummy_tensor = torch.randn(2, 3, 224, 224)
        logits = self.model(dummy_tensor)
        self.assertEqual(logits.shape, (2, len(TARGET_PATHOLOGIES)))

        probs = self.model.predict_probabilities(dummy_tensor)
        self.assertEqual(probs.shape, (2, len(TARGET_PATHOLOGIES)))
        self.assertTrue(torch.all(probs >= 0.0) and torch.all(probs <= 1.0))

    def test_05_monte_carlo_uncertainty(self):
        """Tests stochastic dropout uncertainty sampling and confidence intervals."""
        normal_path = os.path.join(SAMPLE_DIR, "case1_normal.png")
        tensor, _ = preprocess_radiograph(normal_path)

        results = self.uncertainty_engine.sample_predictions(tensor, num_iterations=10)
        self.assertIn("pathology_metrics", results)
        self.assertIn("overall_mean_uncertainty", results)

        for name in TARGET_PATHOLOGIES:
            self.assertIn(name, results["pathology_metrics"])
            p_stats = results["pathology_metrics"][name]
            self.assertGreaterEqual(p_stats["mean_probability"], 0.0)
            self.assertLessEqual(p_stats["mean_probability"], 1.0)
            self.assertGreaterEqual(p_stats["epistemic_uncertainty"], 0.0)
            self.assertLessEqual(p_stats["ci_95_lower"], p_stats["ci_95_upper"])

    def test_06_gradcam_generation(self):
        """Tests Grad-CAM visual heatmap generation and overlay blending."""
        pneumonia_path = os.path.join(SAMPLE_DIR, "case2_pneumonia.png")
        tensor, display_img = preprocess_radiograph(pneumonia_path)

        pneu_idx = TARGET_PATHOLOGIES.index("Pneumonia")
        cam = self.gradcam.generate_heatmap(tensor, pneu_idx)

        self.assertEqual(cam.shape, (224, 224))
        self.assertGreaterEqual(np.min(cam), 0.0)
        self.assertLessEqual(np.max(cam), 1.0)

        blended = self.gradcam.overlay_heatmap(display_img, cam, colormap_name="jet", alpha=0.4)
        self.assertIsInstance(blended, Image.Image)
        self.assertEqual(blended.size, (224, 224))

    def test_07_clinical_report_generation(self):
        """Tests both Markdown and PDF clinical report generation."""
        mock_predictions = {
            "Atelectasis": {"mean_probability": 0.12, "ci_95_lower": 0.08, "ci_95_upper": 0.16, "epistemic_uncertainty": 0.02, "badge": "🟢 Reliable"},
            "Cardiomegaly": {"mean_probability": 0.88, "ci_95_lower": 0.82, "ci_95_upper": 0.94, "epistemic_uncertainty": 0.03, "badge": "🟢 Reliable"},
            "Pleural Effusion": {"mean_probability": 0.15, "ci_95_lower": 0.10, "ci_95_upper": 0.20, "epistemic_uncertainty": 0.025, "badge": "🟢 Reliable"},
            "Pneumonia": {"mean_probability": 0.08, "ci_95_lower": 0.05, "ci_95_upper": 0.11, "epistemic_uncertainty": 0.015, "badge": "🟢 Reliable"},
            "Pneumothorax": {"mean_probability": 0.02, "ci_95_lower": 0.01, "ci_95_upper": 0.04, "epistemic_uncertainty": 0.008, "badge": "🟢 Reliable"},
            "No Finding": {"mean_probability": 0.05, "ci_95_lower": 0.02, "ci_95_upper": 0.08, "epistemic_uncertainty": 0.015, "badge": "🟢 Reliable"}
        }

        mock_ood = {
            "is_valid": True,
            "ood_risk_score": 0.08,
            "reason": "Input verified: Radiograph adheres to thoracic distribution.",
            "metrics": {"contrast_std": 48.2, "dynamic_range": 160.0}
        }

        triage_info = compute_triage_priority(mock_predictions, URGENCY_WEIGHTS)
        self.assertIn("priority_tier", triage_info)
        self.assertEqual(triage_info["primary_concern"], "Cardiomegaly")

        # Markdown test
        md_text = generate_markdown_report("SCAN-TEST-001", mock_predictions, triage_info, mock_ood)
        self.assertIn("CheXGuard AI", md_text)
        self.assertIn("Cardiomegaly", md_text)

        # PDF test
        test_pdf = os.path.join(SAMPLE_DIR, "test_output_report.pdf")
        pdf_path = generate_pdf_report(test_pdf, "SCAN-TEST-001", mock_predictions, triage_info, mock_ood)
        self.assertTrue(os.path.exists(pdf_path))
        self.assertGreater(os.path.getsize(pdf_path), 1000)

        # Cleanup
        if os.path.exists(test_pdf):
            os.remove(test_pdf)


if __name__ == "__main__":
    unittest.main()
