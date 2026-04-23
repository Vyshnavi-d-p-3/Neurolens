"""Shared evaluation metrics for attacks and transfer."""

from __future__ import annotations

import torch
import torch.nn as nn

from attacks.base import Attack
from attacks.fgsm import FGSM
from attacks.pgd import PGD


@torch.no_grad()
def clean_accuracy(model: nn.Module, images: torch.Tensor, labels: torch.Tensor) -> float:
    model.eval()
    preds = model(images).argmax(dim=-1)
    return (preds == labels).float().mean().item()


def attack_success_rate(
    model: nn.Module,
    attack: Attack,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Fraction of samples misclassified after attack (higher = stronger attack)."""
    model.eval()
    with torch.enable_grad():
        x_adv = attack.perturb(images, labels)
    with torch.no_grad():
        preds = model(x_adv).argmax(dim=-1)
        return (preds != labels).float().mean().item()


def evaluate_asr_curve(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilons: list[float],
    attack_name: str = "pgd",
    pgd_steps: int = 20,
) -> dict[float, float]:
    """ASR at each epsilon for FGSM or PGD."""
    results: dict[float, float] = {}
    for eps in epsilons:
        if attack_name == "fgsm":
            attack: Attack = FGSM(model, epsilon=eps)
        else:
            attack = PGD(model, epsilon=eps, steps=pgd_steps)
        results[eps] = attack_success_rate(model, attack, images, labels)
    return results
