"""Lightweight training loops for MVP models."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.dual_encoder import CLIPLite
from models.resnet import ResNet18


@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


def train_resnet(
    model: ResNet18,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 3,
    lr: float = 0.1,
) -> dict[str, list[float]]:
    """Short SGD training for CIFAR-10 MVP."""
    model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    history: dict[str, list[float]] = {"train_loss": [], "test_acc": []}

    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        n = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * labels.size(0)
            n += labels.size(0)
        scheduler.step()
        history["train_loss"].append(epoch_loss / max(n, 1))
        history["test_acc"].append(accuracy(model, test_loader, device))

    return history


def train_clip_lite(
    model: CLIPLite,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 3,
    lr: float = 1e-3,
) -> dict[str, list[float]]:
    """Contrastive training for CLIP-lite on (image, class) pairs."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: dict[str, list[float]] = {"train_loss": [], "retrieval_acc": []}

    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        n = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = model.contrastive_loss(images, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * labels.size(0)
            n += labels.size(0)
        history["train_loss"].append(epoch_loss / max(n, 1))

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                correct += int(model.retrieval_accuracy(images, labels) * labels.size(0))
                total += labels.size(0)
        history["retrieval_acc"].append(correct / max(total, 1))

    return history
