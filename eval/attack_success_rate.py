"""
Attack Success Rate (ASR) vs perturbation budget ε.

MVP: evaluate FGSM/PGD on a held-out CIFAR-10 batch and print ASR curve.
"""

from __future__ import annotations

import argparse

import torch

from eval.metrics import evaluate_asr_curve
from models.resnet import ResNet18
from utils.data import get_cifar10_loaders, get_device


def collect_batch(loader, max_samples: int, device: torch.device):
    images_list, labels_list = [], []
    for batch_images, batch_labels in loader:
        images_list.append(batch_images)
        labels_list.append(batch_labels)
        if sum(t.size(0) for t in images_list) >= max_samples:
            break
    images = torch.cat(images_list, dim=0)[:max_samples].to(device)
    labels = torch.cat(labels_list, dim=0)[:max_samples].to(device)
    return images, labels


def main():
    parser = argparse.ArgumentParser(description="Attack success rate vs epsilon")
    parser.add_argument("--epsilon", type=float, nargs="+", default=[0.01, 0.03, 0.05, 0.1, 0.2])
    parser.add_argument("--attack", choices=["fgsm", "pgd"], default="pgd")
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/resnet18_mvp.pt")
    args = parser.parse_args()

    device = get_device()
    _, test_loader = get_cifar10_loaders(batch_size=128, test_subset=args.num_samples)

    model = ResNet18().to(device)
    try:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"Checkpoint not found at {args.checkpoint}. Run: python run_mvp.py")

    images, labels = collect_batch(test_loader, args.num_samples, device)
    curve = evaluate_asr_curve(model, images, labels, args.epsilon, attack_name=args.attack)

    print(f"\nAttack Success Rate ({args.attack.upper()})")
    print("-" * 32)
    for eps, asr in curve.items():
        print(f"  ε={eps:.2f}  ASR={asr:.1%}")


if __name__ == "__main__":
    main()
