"""
Cross-Modal Perturbation Transfer Attack (Novel contribution).

Hypothesis: Perturbations crafted against a multimodal model (CLIP-lite)
transfer to standalone unimodal classifiers because both share similar
low-level feature representations.

Method:
1. Given image x and correct caption c, find δ that minimizes
   similarity(encode_image(x+δ), encode_text(c)) while maximizing
   similarity with a random incorrect caption.
2. Apply δ to standalone ResNet-18 (NOT used to craft δ).
3. Measure transfer rate: % of images where perturbation fools ResNet-18.
4. Compare to PGD crafted directly against ResNet-18 (upper bound).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from attacks.base import Attack


class CrossModalTransfer(Attack):
    """
    Craft adversarial perturbation against CLIP-lite, measure transfer to classifier.

    Usage:
        attack = CrossModalTransfer(
            clip_model=clip_lite,
            target_model=resnet18,
            epsilon=0.1,
        )
        x_adv = attack.perturb_clip(images, correct_captions, wrong_captions)
        transfer_rate = attack.measure_transfer(x_adv, labels)
    """

    def __init__(
        self,
        model: nn.Module,  # CLIP-lite model
        epsilon: float = 0.1,
        steps: int = 20,
        step_size: float | None = None,
    ):
        super().__init__(model, epsilon)
        self.steps = steps
        self.step_size = step_size or (epsilon / 4)
        self._target_model: nn.Module | None = None

    def set_target_model(self, target: nn.Module):
        """Set the standalone classifier to measure transfer against."""
        self._target_model = target
        self._target_model.eval()

    def perturb(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Standard interface — not applicable for cross-modal. Use perturb_clip instead."""
        raise NotImplementedError("Use perturb_clip() for cross-modal attack")

    def perturb_clip(
        self,
        images: torch.Tensor,
        correct_text_embeddings: torch.Tensor,
        incorrect_text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Craft adversarial images against CLIP-lite's contrastive objective.

        Minimizes similarity with correct caption, maximizes with incorrect.
        This is PGD-style optimization against the contrastive loss.
        """
        x_adv = images.clone().detach()
        x_adv += torch.zeros_like(x_adv).uniform_(-self.epsilon, self.epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        for _ in range(self.steps):
            x_adv.requires_grad_(True)

            # Get image embeddings from CLIP-lite
            img_emb = self.model.encode_image(x_adv)
            img_emb = F.normalize(img_emb, dim=-1)

            # Contrastive loss: push away from correct, pull toward incorrect
            correct_sim = (img_emb * correct_text_embeddings).sum(dim=-1)
            incorrect_sim = (img_emb * incorrect_text_embeddings).sum(dim=-1)
            loss = correct_sim - incorrect_sim  # minimize correct, maximize incorrect

            loss.sum().backward()

            with torch.no_grad():
                x_adv = x_adv - self.step_size * x_adv.grad.sign()  # gradient descent on loss
                x_adv = self._clamp(x_adv, images)

            x_adv = x_adv.detach()

        return x_adv

    def measure_transfer(
        self,
        x_adv: torch.Tensor,
        true_labels: torch.Tensor,
    ) -> float:
        """
        Measure transfer rate: what fraction of CLIP-crafted adversarial
        examples also fool the standalone classifier?

        Returns:
            Transfer rate as float in [0, 1].
        """
        if self._target_model is None:
            raise ValueError("Call set_target_model() first")

        with torch.no_grad():
            preds = self._target_model(x_adv).argmax(dim=-1)
            fooled = (preds != true_labels).float()
            return fooled.mean().item()
