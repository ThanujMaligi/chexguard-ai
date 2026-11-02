"""
Monte Carlo Dropout & Epistemic Uncertainty Quantification Module for CheXGuard AI.
Implements Bayesian approximation via active dropout sampling to provide
statistically grounded confidence intervals for each thoracic pathology.
"""

from typing import Dict, List, Any
import numpy as np
import torch
import torch.nn as nn


class MonteCarloUncertaintyEngine:
    """
    Evaluates epistemic uncertainty and statistical risk bounds using
    stochastic Monte Carlo forward passes through dropout layers.
    """

    def __init__(self, model: nn.Module, pathology_names: List[str]):
        self.model = model
        self.pathology_names = pathology_names

    def _enable_dropout(self):
        """Forces dropout layers into training mode while keeping batchnorm frozen."""
        for m in self.model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.train()

    def sample_predictions(
        self,
        input_tensor: torch.Tensor,
        num_iterations: int = 15
    ) -> Dict[str, Any]:
        """
        Executes stochastic forward passes to compute posterior distribution statistics.
        
        Args:
            input_tensor: Image tensor of shape [1, 3, H, W]
            num_iterations: Number of Monte Carlo samples (default 15)
            
        Returns:
            Dictionary containing:
                - mean_probabilities: Mean prediction per pathology
                - epistemic_uncertainties: Standard deviations (uncertainty measure)
                - confidence_intervals_95: (lower, upper) 95% CI bounds
                - entropy: Shannon predictive entropy
                - reliability_flags: Categorical safety flag per condition
                - raw_samples: Array of shape [num_iterations, num_classes]
        """
        self.model.eval()
        self._enable_dropout()

        samples = []
        with torch.no_grad():
            if hasattr(self.model, 'forward_from_features'):
                # Accelerated Bayesian sampling: extract deterministic features once
                feat = self.model.features(input_tensor)
                feat = torch.nn.functional.relu(feat, inplace=True)
                feat_norm = torch.nn.functional.normalize(feat, p=2, dim=1)
                anat = self.model.extract_anatomical_features(input_tensor)
                for _ in range(num_iterations):
                    logits = self.model.forward_from_features(feat_norm, anat)
                    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                    samples.append(probs)
            else:
                for _ in range(num_iterations):
                    logits = self.model(input_tensor)
                    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                    samples.append(probs)

        # Revert model to evaluation mode
        self.model.eval()

        samples_arr = np.array(samples)  # Shape: [T, num_classes]
        means = np.mean(samples_arr, axis=0)
        stds = np.std(samples_arr, axis=0)

        # 95% Confidence Intervals
        ci_lower = np.clip(means - 1.96 * stds, 0.0, 1.0)
        ci_upper = np.clip(means + 1.96 * stds, 0.0, 1.0)

        # Shannon Entropy for multi-label binary distribution
        eps = 1e-7
        entropies = - (means * np.log2(means + eps) + (1.0 - means) * np.log2(1.0 - means + eps))

        results = {}
        for idx, name in enumerate(self.pathology_names):
            m = float(means[idx])
            s = float(stds[idx])
            low = float(ci_lower[idx])
            high = float(ci_upper[idx])
            ent = float(entropies[idx])

            if s < 0.05:
                reliability = "HIGH_CONFIDENCE"
                badge = "🟢 Reliable"
            elif s < 0.12:
                reliability = "MODERATE_UNCERTAINTY"
                badge = "🟡 Moderate Variance"
            else:
                reliability = "HIGH_UNCERTAINTY"
                badge = "🔴 Mandatory Specialist Verification"

            results[name] = {
                "mean_probability": round(m, 4),
                "epistemic_uncertainty": round(s, 4),
                "ci_95_lower": round(low, 4),
                "ci_95_upper": round(high, 4),
                "entropy": round(ent, 4),
                "reliability": reliability,
                "badge": badge
            }

        return {
            "pathology_metrics": results,
            "overall_mean_uncertainty": round(float(np.mean(stds)), 4),
            "num_mc_samples": num_iterations
        }
