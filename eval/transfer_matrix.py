"""
Cross-modal transfer evaluation — core MVP result.

Compares:
  1. White-box PGD on ResNet-18 (upper bound)
  2. Cross-modal attack crafted on CLIP-lite, measured on ResNet-18 (transfer)
"""

from __future__ import annotations

import argparse

import torch

from attacks.crossmodal_transfer import CrossModalTransfer
from attacks.pgd import PGD
from eval.metrics import attack_success_rate
from models.dual_encoder import CLIPLite
from models.resnet import ResNet18
from utils.data import get_cifar10_loaders, get_device


def random_wrong_labels(labels: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    wrong = labels.clone()
    for i in range(labels.size(0)):
        if num_classes > 1:
            offset = torch.randint(1, num_classes, (1,)).item()
            wrong[i] = (labels[i] + offset) % num_classes
    return wrong


def run_transfer_eval(
    resnet: ResNet18,
    clip: CLIPLite,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.1,
    pgd_steps: int = 20,
) -> dict[str, float]:
    resnet.eval()
    clip.eval()

    # White-box PGD on ResNet (upper bound)
    pgd = PGD(resnet, epsilon=epsilon, steps=pgd_steps)
    pgd_asr = attack_success_rate(resnet, pgd, images, labels)

    # Cross-modal: craft on CLIP, measure on ResNet
    wrong_labels = random_wrong_labels(labels)
    with torch.no_grad():
        correct_emb = clip.encode_text(labels)
        incorrect_emb = clip.encode_text(wrong_labels)

    cross = CrossModalTransfer(clip, epsilon=epsilon, steps=pgd_steps)
    cross.set_target_model(resnet)
    with torch.enable_grad():
        x_adv = cross.perturb_clip(images, correct_emb, incorrect_emb)
    transfer_asr = cross.measure_transfer(x_adv, labels)

    with torch.no_grad():
        clean_retrieval = clip.retrieval_accuracy(images, labels)
        adv_retrieval = clip.retrieval_accuracy(x_adv, labels)

    return {
        "pgd_resnet_asr": pgd_asr,
        "crossmodal_transfer_asr": transfer_asr,
        "clip_retrieval_clean": clean_retrieval,
        "clip_retrieval_adv": adv_retrieval,
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-modal transfer evaluation")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--resnet-checkpoint", default="checkpoints/resnet18_mvp.pt")
    parser.add_argument("--clip-checkpoint", default="checkpoints/clip_lite_mvp.pt")
    args = parser.parse_args()

    device = get_device()
    _, test_loader = get_cifar10_loaders(batch_size=128, test_subset=args.num_samples)

    images_list, labels_list = [], []
    for img, lab in test_loader:
        images_list.append(img)
        labels_list.append(lab)
    images = torch.cat(images_list, dim=0).to(device)
    labels = torch.cat(labels_list, dim=0).to(device)

    resnet = ResNet18().to(device)
    clip = CLIPLite().to(device)
    resnet.load_state_dict(torch.load(args.resnet_checkpoint, map_location=device, weights_only=True))
    clip.load_state_dict(torch.load(args.clip_checkpoint, map_location=device, weights_only=True))

    results = run_transfer_eval(resnet, clip, images, labels, epsilon=args.epsilon)

    print("\nCross-Modal Transfer Results (ε={:.2f})".format(args.epsilon))
    print("=" * 48)
    print(f"  PGD on ResNet-18 (white-box):     {results['pgd_resnet_asr']:.1%}")
    print(f"  CLIP→ResNet transfer:             {results['crossmodal_transfer_asr']:.1%}")
    print(f"  CLIP retrieval (clean / adv):     {results['clip_retrieval_clean']:.1%} / {results['clip_retrieval_adv']:.1%}")
    transfer_ratio = (
        results["crossmodal_transfer_asr"] / results["pgd_resnet_asr"]
        if results["pgd_resnet_asr"] > 0 else 0.0
    )
    print(f"  Transfer / white-box ratio:       {transfer_ratio:.1%}")


if __name__ == "__main__":
    main()
