#!/usr/bin/env python3
"""
NeuroLens MVP — train models, run attacks, report cross-modal transfer.

Usage:
    python run_mvp.py              # full MVP (~few min on CPU with --quick)
    python run_mvp.py --quick      # fast smoke run for CI / demos
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from eval.metrics import evaluate_asr_curve
from eval.transfer_matrix import run_transfer_eval
from models.dual_encoder import CLIPLite
from models.resnet import ResNet18
from utils.data import get_cifar10_loaders, get_device
from utils.training import train_clip_lite, train_resnet


def save_checkpoint(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def main():
    parser = argparse.ArgumentParser(description="NeuroLens MVP pipeline")
    parser.add_argument("--quick", action="store_true", help="Fast run: small data, few epochs")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--eval-samples", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=0.1)
    args = parser.parse_args()

    if args.quick:
        epochs = args.epochs or 3
        train_samples = args.train_samples or 5000
        eval_samples = args.eval_samples or 256
    else:
        epochs = args.epochs or 5
        train_samples = args.train_samples or 10000
        eval_samples = args.eval_samples or 512

    device = get_device()
    ckpt_dir = Path("checkpoints")
    print(f"Device: {device}")
    print(f"Training on {train_samples} samples, {epochs} epochs\n")

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=128,
        train_subset=train_samples,
        test_subset=eval_samples,
    )

    # --- Train ResNet-18 ---
    print("=" * 50)
    print("1. Training ResNet-18 on CIFAR-10")
    print("=" * 50)
    resnet = ResNet18().to(device)
    resnet_hist = train_resnet(resnet, train_loader, test_loader, device, epochs=epochs)
    resnet_path = ckpt_dir / "resnet18_mvp.pt"
    save_checkpoint(resnet, resnet_path)
    print(f"   Final test accuracy: {resnet_hist['test_acc'][-1]:.1%}")
    print(f"   Saved → {resnet_path}\n")

    # --- Train CLIP-lite ---
    print("=" * 50)
    print("2. Training CLIP-lite (contrastive)")
    print("=" * 50)
    clip = CLIPLite().to(device)
    clip_hist = train_clip_lite(clip, train_loader, test_loader, device, epochs=epochs)
    clip_path = ckpt_dir / "clip_lite_mvp.pt"
    save_checkpoint(clip, clip_path)
    print(f"   Final retrieval accuracy: {clip_hist['retrieval_acc'][-1]:.1%}")
    print(f"   Saved → {clip_path}\n")

    # --- Collect eval batch ---
    images_list, labels_list = [], []
    for img, lab in test_loader:
        images_list.append(img)
        labels_list.append(lab)
    images = torch.cat(images_list, dim=0).to(device)
    labels = torch.cat(labels_list, dim=0).to(device)

    # --- ASR curve ---
    print("=" * 50)
    print("3. Attack Success Rate (PGD on ResNet-18)")
    print("=" * 50)
    epsilons = [0.03, 0.05, 0.1, 0.2] if args.quick else [0.01, 0.03, 0.05, 0.1, 0.2]
    asr_curve = evaluate_asr_curve(resnet, images, labels, epsilons, attack_name="pgd", pgd_steps=10 if args.quick else 20)
    for eps, asr in asr_curve.items():
        print(f"   ε={eps:.2f}  ASR={asr:.1%}")

    # --- Cross-modal transfer (key result) ---
    print("\n" + "=" * 50)
    print("4. Cross-Modal Transfer (CLIP → ResNet)")
    print("=" * 50)
    transfer = run_transfer_eval(
        resnet, clip, images, labels,
        epsilon=args.epsilon,
        pgd_steps=10 if args.quick else 20,
    )
    print(f"   PGD on ResNet (white-box):  {transfer['pgd_resnet_asr']:.1%}")
    print(f"   CLIP→ResNet transfer:       {transfer['crossmodal_transfer_asr']:.1%}")
    print(f"   CLIP retrieval clean→adv:   {transfer['clip_retrieval_clean']:.1%} → {transfer['clip_retrieval_adv']:.1%}")

    if transfer["pgd_resnet_asr"] > 0:
        ratio = transfer["crossmodal_transfer_asr"] / transfer["pgd_resnet_asr"]
        print(f"\n   Transfer efficacy: {ratio:.0%} of white-box PGD")

    print("\n" + "=" * 50)
    print("MVP complete. Next steps:")
    print("  python -m eval.attack_success_rate")
    print("  python -m eval.transfer_matrix")
    print("  streamlit run demo/app.py")
    print("=" * 50)


if __name__ == "__main__":
    main()
