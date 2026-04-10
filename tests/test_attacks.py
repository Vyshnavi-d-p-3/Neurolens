"""Tests for attack implementations — gradient correctness and sanity checks."""

import torch
import pytest
from models.resnet import ResNet18
from attacks.fgsm import FGSM
from attacks.pgd import PGD


@pytest.fixture
def model():
    m = ResNet18(num_classes=10)
    m.eval()
    return m


@pytest.fixture
def sample_batch():
    images = torch.rand(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 3])
    return images, labels


class TestFGSM:
    def test_epsilon_zero_no_change(self, model, sample_batch):
        """FGSM with ε=0 should not change the input."""
        images, labels = sample_batch
        attack = FGSM(model, epsilon=0.0)
        x_adv = attack.perturb(images, labels)
        assert torch.allclose(x_adv, images, atol=1e-7)

    def test_output_in_valid_range(self, model, sample_batch):
        """Adversarial examples should be in [0, 1]."""
        images, labels = sample_batch
        attack = FGSM(model, epsilon=0.1)
        x_adv = attack.perturb(images, labels)
        assert x_adv.min() >= 0.0
        assert x_adv.max() <= 1.0

    def test_perturbation_bounded(self, model, sample_batch):
        """Perturbation should be within ε-ball."""
        images, labels = sample_batch
        eps = 0.1
        attack = FGSM(model, epsilon=eps)
        x_adv = attack.perturb(images, labels)
        assert (x_adv - images).abs().max() <= eps + 1e-6


class TestPGD:
    def test_output_in_valid_range(self, model, sample_batch):
        images, labels = sample_batch
        attack = PGD(model, epsilon=0.1, steps=5)
        x_adv = attack.perturb(images, labels)
        assert x_adv.min() >= 0.0
        assert x_adv.max() <= 1.0

    def test_perturbation_bounded(self, model, sample_batch):
        images, labels = sample_batch
        eps = 0.1
        attack = PGD(model, epsilon=eps, steps=5)
        x_adv = attack.perturb(images, labels)
        assert (x_adv - images).abs().max() <= eps + 1e-6

    def test_stronger_than_fgsm(self, model, sample_batch):
        """PGD should produce larger loss than FGSM (or equal)."""
        images, labels = sample_batch
        eps = 0.3
        criterion = torch.nn.CrossEntropyLoss()

        fgsm_adv = FGSM(model, eps).perturb(images, labels)
        pgd_adv = PGD(model, eps, steps=10).perturb(images, labels)

        fgsm_loss = criterion(model(fgsm_adv), labels).item()
        pgd_loss = criterion(model(pgd_adv), labels).item()

        # PGD should find at least as high a loss as FGSM
        assert pgd_loss >= fgsm_loss - 0.1  # small tolerance
