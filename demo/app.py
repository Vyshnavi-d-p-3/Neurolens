"""
NeuroLens Streamlit demo — visualize adversarial attacks and cross-modal transfer.

Run: streamlit run demo/app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import torch
import torchvision
from torchvision.utils import make_grid

from attacks.crossmodal_transfer import CrossModalTransfer
from attacks.fgsm import FGSM
from attacks.pgd import PGD
from eval.transfer_matrix import random_wrong_labels, run_transfer_eval
from models.dual_encoder import CLIPLite, CIFAR10_CLASSES
from models.resnet import ResNet18
st.set_page_config(page_title="NeuroLens MVP", page_icon="🧠", layout="wide")

st.title("NeuroLens MVP")
st.caption("Multimodal adversarial robustness — cross-modal perturbation transfer")


@st.cache_resource
def load_models(device_str: str):
    device = torch.device(device_str)
    resnet = ResNet18().to(device)
    clip = CLIPLite().to(device)
    resnet_path = Path("checkpoints/resnet18_mvp.pt")
    clip_path = Path("checkpoints/clip_lite_mvp.pt")
    if resnet_path.exists():
        resnet.load_state_dict(torch.load(resnet_path, map_location=device, weights_only=True))
    if clip_path.exists():
        clip.load_state_dict(torch.load(clip_path, map_location=device, weights_only=True))
    resnet.eval()
    clip.eval()
    return resnet, clip, device


def tensor_to_grid(images: torch.Tensor, nrow: int = 8) -> torch.Tensor:
    return make_grid(images.cpu(), nrow=nrow, padding=2, normalize=False)


device_choice = st.sidebar.selectbox("Device", ["cpu", "cuda", "mps"], index=0)
if device_choice == "cuda" and not torch.cuda.is_available():
    st.sidebar.warning("CUDA unavailable, using CPU")
    device_choice = "cpu"
if device_choice == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
    device_choice = "cpu"

resnet, clip, device = load_models(device_choice)

ckpt_ok = Path("checkpoints/resnet18_mvp.pt").exists()
if not ckpt_ok:
    st.warning("No checkpoints found. Run `python run_mvp.py --quick` first.")
    st.stop()

epsilon = st.sidebar.slider("Perturbation ε", 0.01, 0.3, 0.1, 0.01)
attack_type = st.sidebar.radio("Attack", ["FGSM", "PGD", "Cross-Modal (CLIP→ResNet)"])
num_images = st.sidebar.slider("Images", 4, 32, 8)

# Sample CIFAR-10 test images
dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
indices = torch.randperm(len(dataset))[:num_images].tolist()
images = torch.stack([dataset[i][0] for i in indices])
labels = torch.tensor([dataset[i][1] for i in indices]).to(device)
images_dev = images.to(device)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Clean")
    st.image(tensor_to_grid(images).permute(1, 2, 0).numpy() / 255.0, use_container_width=True)
    for i, lab in enumerate(labels.cpu().tolist()):
        st.caption(f"#{i}: {CIFAR10_CLASSES[lab]}")

with torch.no_grad():
    if attack_type == "FGSM":
        x_adv = FGSM(resnet, epsilon=epsilon).perturb(images_dev, labels)
    elif attack_type == "PGD":
        x_adv = PGD(resnet, epsilon=epsilon, steps=20).perturb(images_dev, labels)
    else:
        wrong = random_wrong_labels(labels)
        cross = CrossModalTransfer(clip, epsilon=epsilon, steps=20)
        x_adv = cross.perturb_clip(
            images_dev,
            clip.encode_text(labels),
            clip.encode_text(wrong),
        )

    preds_clean = resnet(images_dev).argmax(dim=-1)
    preds_adv = resnet(x_adv).argmax(dim=-1)

with col2:
    st.subheader("Adversarial")
    st.image(tensor_to_grid(x_adv).permute(1, 2, 0).numpy() / 255.0, use_container_width=True)
    fooled = (preds_adv != labels).sum().item()
    st.metric("ResNet fooled", f"{fooled}/{num_images}")

st.divider()
st.subheader("Transfer comparison")
if st.button("Run transfer eval on batch"):
    with st.spinner("Evaluating..."):
        results = run_transfer_eval(resnet, clip, images_dev, labels, epsilon=epsilon)
    c1, c2, c3 = st.columns(3)
    c1.metric("PGD white-box ASR", f"{results['pgd_resnet_asr']:.0%}")
    c2.metric("CLIP→ResNet transfer", f"{results['crossmodal_transfer_asr']:.0%}")
    c3.metric("CLIP retrieval drop", f"{results['clip_retrieval_clean']:.0%} → {results['clip_retrieval_adv']:.0%}")

st.markdown(
    """
**Research hypothesis:** Perturbations optimized against CLIP-lite's contrastive objective
transfer to standalone ResNet-18 because both models share low-level visual features.
"""
)
