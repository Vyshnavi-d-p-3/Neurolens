"""Tests for the PGD adversarial training defense."""

import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset

from models.resnet import ResNet18
from defenses.adversarial_training import AdversarialTrainer


@pytest.fixture
def loader():
    """Small synthetic CIFAR-shaped dataset."""
    torch.manual_seed(0)
    x = torch.rand(32, 3, 32, 32)
    y = torch.randint(0, 10, (32,))
    return DataLoader(TensorDataset(x, y), batch_size=16)


@pytest.fixture
def trainer():
    model = ResNet18(num_classes=10)
    return AdversarialTrainer(model, epsilon=0.1, pgd_steps=3,
                              device=torch.device("cpu"))


class TestAdversarialTrainer:
    def test_pgd_perturbation_bounded(self, trainer, loader):
        """Training adversarials stay within the ε-ball and [0, 1]."""
        images, labels = next(iter(loader))
        x_adv = trainer._pgd_attack(images, labels)
        assert (x_adv - images).abs().max() <= trainer.epsilon + 1e-6
        assert x_adv.min() >= 0.0 and x_adv.max() <= 1.0

    def test_train_epoch_returns_metrics(self, trainer, loader):
        """train_epoch reports the four expected scalars."""
        opt = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
        stats = trainer.train_epoch(loader, opt)
        assert set(stats) == {
            "clean_loss", "adv_loss", "train_clean_acc", "train_adv_acc"
        }
        for v in stats.values():
            assert isinstance(v, float)

    def test_training_changes_weights(self, trainer, loader):
        """A training step must update model parameters."""
        before = [p.clone() for p in trainer.model.parameters()]
        opt = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
        trainer.train_epoch(loader, opt)
        after = list(trainer.model.parameters())
        assert any(not torch.equal(b, a) for b, a in zip(before, after))

    def test_robust_accuracy_is_a_fraction(self, trainer, loader):
        """evaluate_robust returns a value in [0, 1]."""
        acc = trainer.evaluate_robust(loader, pgd_steps=5)
        assert 0.0 <= acc <= 1.0

    def test_robust_not_above_clean(self, trainer, loader):
        """Robust accuracy cannot exceed clean accuracy for the same model."""
        clean = trainer.evaluate_clean(loader)
        robust = trainer.evaluate_robust(loader, pgd_steps=5)
        assert robust <= clean + 1e-6
