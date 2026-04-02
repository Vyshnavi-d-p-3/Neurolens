"""
Randomized Smoothing — certified adversarial robustness.

From: Cohen et al., "Certified Adversarial Robustness via Randomized Smoothing" (2019).

The only scalable certified defense. Wraps a base classifier in Gaussian noise,
then uses the Neyman-Pearson lemma to compute a certified radius — if the top
class probability exceeds a threshold, no L2 perturbation within that radius
can change the prediction.

Certification via Monte Carlo sampling (N=1000 per input).
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
from scipy.stats import norm


class RandomizedSmoothing:
    """
    Certified defense via randomized smoothing.

    Usage:
        smoother = RandomizedSmoothing(base_model, sigma=0.25, n_samples=1000)
        pred, radius = smoother.certify(image)
        # pred is the smoothed prediction
        # radius is the certified L2 radius (no attack within this radius can change pred)
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_classes: int = 10,
        sigma: float = 0.25,
        n_samples: int = 1000,
        alpha: float = 0.001,
    ):
        self.base_model = base_model
        self.num_classes = num_classes
        self.sigma = sigma
        self.n_samples = n_samples
        self.alpha = alpha  # confidence level for Clopper-Pearson
        self.base_model.eval()

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Smoothed prediction: majority vote over n_samples noisy copies.

        Returns predicted class for each input in the batch.
        """
        counts = self._sample_noise(x)
        return counts.argmax(dim=-1)

    @torch.no_grad()
    def certify(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Certify robustness for each input.

        Returns:
            predictions: Predicted class per input (batch_size,)
            radii: Certified L2 radius per input (batch_size,)
                   Radius = 0.0 means certification failed (abstain).
        """
        batch_size = x.shape[0]
        counts = self._sample_noise(x)

        predictions = counts.argmax(dim=-1)
        radii = torch.zeros(batch_size, device=x.device)

        for i in range(batch_size):
            top_count = counts[i].max().item()
            total = counts[i].sum().item()

            # Clopper-Pearson lower bound on p_A (probability of top class)
            p_lower = self._clopper_pearson_lower(top_count, total, self.alpha)

            if p_lower > 0.5:
                # Certified radius via Neyman-Pearson lemma
                radii[i] = self.sigma * norm.ppf(p_lower)
            # else: abstain (radius = 0)

        return predictions, radii

    def _sample_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sample n_samples noisy copies and count class predictions.

        Returns counts tensor (batch_size, num_classes).
        """
        batch_size = x.shape[0]
        counts = torch.zeros(batch_size, self.num_classes, device=x.device)

        # Process in batches to avoid OOM
        samples_per_batch = min(self.n_samples, 100)
        num_batches = math.ceil(self.n_samples / samples_per_batch)

        for _ in range(num_batches):
            n = min(samples_per_batch, self.n_samples)

            # Add Gaussian noise: repeat input n times, add noise
            x_noisy = x.repeat(n, 1, 1, 1)
            noise = torch.randn_like(x_noisy) * self.sigma
            x_noisy = x_noisy + noise

            # Classify all noisy samples
            logits = self.base_model(x_noisy)
            preds = logits.argmax(dim=-1)

            # Count predictions per original input
            preds = preds.view(n, batch_size)
            for j in range(batch_size):
                for pred in preds[:, j]:
                    counts[j, pred.item()] += 1

        return counts

    @staticmethod
    def _clopper_pearson_lower(successes: int, total: int, alpha: float) -> float:
        """Clopper-Pearson lower bound for binomial proportion."""
        from scipy.stats import beta as beta_dist

        if successes == 0:
            return 0.0
        return beta_dist.ppf(alpha / 2, successes, total - successes + 1)
