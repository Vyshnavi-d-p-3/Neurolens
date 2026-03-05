"""Abstract base class for adversarial attacks."""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class Attack(ABC):
    """
    Base interface for all adversarial attacks.

    Subclasses implement perturb() which takes a model, clean input, and
    true label, and returns an adversarial example.
    """

    def __init__(self, model: nn.Module, epsilon: float):
        self.model = model
        self.epsilon = epsilon
        self.model.eval()

    @abstractmethod
    def perturb(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Generate adversarial example.

        Args:
            x: Clean input tensor (batch_size, C, H, W)
            y: True labels (batch_size,)

        Returns:
            x_adv: Adversarial input, same shape as x, within ε-ball of x
        """
        ...

    def _clamp(self, x_adv: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Clamp adversarial example to ε-ball around x and valid pixel range [0,1]."""
        delta = torch.clamp(x_adv - x, min=-self.epsilon, max=self.epsilon)
        return torch.clamp(x + delta, 0.0, 1.0)
