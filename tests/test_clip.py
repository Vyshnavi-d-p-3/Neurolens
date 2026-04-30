"""Tests for CLIP-lite and cross-modal transfer."""

import torch
import pytest

from models.dual_encoder import CLIPLite
from models.resnet import ResNet18
from attacks.crossmodal_transfer import CrossModalTransfer
from attacks.pgd import PGD


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def batch(device):
    images = torch.rand(4, 3, 32, 32, device=device)
    labels = torch.tensor([0, 1, 2, 3], device=device)
    return images, labels


class TestCLIPLite:
    def test_encode_shapes(self, device, batch):
        images, labels = batch
        clip = CLIPLite(embed_dim=64).to(device)
        img_emb = clip.encode_image(images)
        txt_emb = clip.encode_text(labels)
        assert img_emb.shape == (4, 64)
        assert txt_emb.shape == (4, 64)
        assert torch.allclose(img_emb.norm(dim=-1), torch.ones(4, device=device), atol=1e-5)

    def test_contrastive_loss_finite(self, device, batch):
        images, labels = batch
        clip = CLIPLite().to(device)
        loss = clip.contrastive_loss(images, labels)
        assert loss.ndim == 0
        assert torch.isfinite(loss)


class TestCrossModalTransfer:
    def test_perturbation_bounded(self, device, batch):
        images, labels = batch
        clip = CLIPLite().to(device)
        wrong = (labels + 1) % 10
        attack = CrossModalTransfer(clip, epsilon=0.1, steps=3)
        x_adv = attack.perturb_clip(
            images,
            clip.encode_text(labels),
            clip.encode_text(wrong),
        )
        assert (x_adv - images).abs().max() <= 0.1 + 1e-6

    def test_measure_transfer(self, device, batch):
        images, labels = batch
        clip = CLIPLite().to(device)
        resnet = ResNet18().to(device)
        wrong = (labels + 1) % 10
        attack = CrossModalTransfer(clip, epsilon=0.2, steps=5)
        attack.set_target_model(resnet)
        x_adv = attack.perturb_clip(
            images,
            clip.encode_text(labels),
            clip.encode_text(wrong),
        )
        rate = attack.measure_transfer(x_adv, labels)
        assert 0.0 <= rate <= 1.0

    def test_pgd_stronger_or_equal_transfer_setup(self, device, batch):
        """Sanity: PGD white-box runs without error alongside cross-modal."""
        images, labels = batch
        resnet = ResNet18().to(device)
        pgd = PGD(resnet, epsilon=0.1, steps=3)
        x_adv = pgd.perturb(images, labels)
        assert x_adv.shape == images.shape
