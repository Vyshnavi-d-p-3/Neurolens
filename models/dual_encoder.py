"""
CLIP-lite: minimal dual encoder for cross-modal adversarial research.

Image encoder reuses ResNet-18 features (from scratch). Text side uses learned
class-caption embeddings on CIFAR-10 (label as caption) for a simple, reproducible MVP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet import ResNet18

CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


class CLIPLite(nn.Module):
    """
    Contrastive image–text encoder for MVP experiments.

    - encode_image(x) -> L2-normalized embedding
    - encode_text(class_ids) -> L2-normalized embedding (learned per-class captions)
    """

    def __init__(self, embed_dim: int = 128, num_classes: int = 10, temperature: float = 0.07):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.temperature = temperature

        backbone = ResNet18(num_classes=num_classes)
        self.image_backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,
        )
        self.image_proj = nn.Linear(512, embed_dim)
        self.text_embedding = nn.Embedding(num_classes, embed_dim)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.image_backbone(x)
        feats = torch.flatten(feats, 1)
        return F.normalize(self.image_proj(feats), dim=-1)

    def encode_text(self, class_ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text_embedding(class_ids), dim=-1)

    def contrastive_loss(
        self,
        images: torch.Tensor,
        class_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Symmetric InfoNCE loss (batch diagonal as positives)."""
        img_emb = self.encode_image(images)
        txt_emb = self.encode_text(class_ids)
        logits = img_emb @ txt_emb.T / self.temperature
        labels = torch.arange(images.size(0), device=images.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2

    @torch.no_grad()
    def retrieval_accuracy(self, images: torch.Tensor, class_ids: torch.Tensor) -> float:
        """Fraction where nearest text embedding matches the image label."""
        img_emb = self.encode_image(images)
        txt_emb = self.encode_text(torch.arange(self.num_classes, device=images.device))
        sims = img_emb @ txt_emb.T
        preds = sims.argmax(dim=-1)
        return (preds == class_ids).float().mean().item()
