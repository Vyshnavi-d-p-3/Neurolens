"""
Fast Gradient Sign Method (FGSM).

From: Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (2015).

Single-step attack: x_adv = x + ε · sign(∇_x L(θ, x, y))

The gradient of the loss with respect to the input tells us which direction
in input space increases the loss the most. FGSM takes one step of size ε
in that direction.
"""

import torch
import torch.nn as nn

from attacks.base import Attack


class FGSM(Attack):
    """
    FGSM: one gradient step in the sign direction.

    Usage:
        attack = FGSM(model, epsilon=0.1)
        x_adv = attack.perturb(images, labels)
    """

    def __init__(self, model: nn.Module, epsilon: float = 0.1):
        super().__init__(model, epsilon)

    def perturb(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_adv = x.clone().detach().requires_grad_(True)

        # Forward pass
        logits = self.model(x_adv)
        loss = nn.CrossEntropyLoss()(logits, y)

        # Backward pass — compute gradient of loss w.r.t. input
        loss.backward()

        # FGSM step: x_adv = x + ε · sign(∇_x L)
        with torch.no_grad():
            perturbation = self.epsilon * x_adv.grad.sign()
            x_adv = x + perturbation
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        return x_adv.detach()
