"""
Projected Gradient Descent (PGD) attack.

From: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks" (2018).

Iterative attack: x_{t+1} = Π_{B(x,ε)}(x_t + α · sign(∇_x L))

PGD is iterated FGSM with projection back to the ε-ball after each step,
starting from a random point within the ball. This is the standard
"strongest first-order attack" used for adversarial training evaluation.
"""

import torch
import torch.nn as nn

from attacks.base import Attack


class PGD(Attack):
    """
    PGD-k: k steps of projected gradient ascent on the loss.

    Default: PGD-20 with step_size = ε/4, random start.

    Usage:
        attack = PGD(model, epsilon=0.1, steps=20)
        x_adv = attack.perturb(images, labels)
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.1,
        steps: int = 20,
        step_size: float | None = None,
        random_start: bool = True,
    ):
        super().__init__(model, epsilon)
        self.steps = steps
        self.step_size = step_size or (epsilon / 4)
        self.random_start = random_start

    def perturb(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_adv = x.clone().detach()

        # Random initialization within ε-ball
        if self.random_start:
            x_adv = x_adv + torch.zeros_like(x_adv).uniform_(-self.epsilon, self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        for _ in range(self.steps):
            x_adv.requires_grad_(True)

            # Forward + backward
            logits = self.model(x_adv)
            loss = nn.CrossEntropyLoss()(logits, y)
            loss.backward()

            with torch.no_grad():
                # Gradient ascent step
                x_adv = x_adv + self.step_size * x_adv.grad.sign()

                # Project back to ε-ball around original x
                x_adv = self._clamp(x_adv, x)

            x_adv = x_adv.detach()

        return x_adv
