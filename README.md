<div align="center">

# NeuroLens

**Multimodal Adversarial Robustness Toolkit & Research**

[![CI](https://github.com/Vyshnavi-d-p-3/neurolens/actions/workflows/ci.yml/badge.svg)](https://github.com/Vyshnavi-d-p-3/neurolens/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*From-scratch model implementations, adversarial attacks from original papers,*
*defenses with certified guarantees, and a novel cross-modal perturbation transfer study.*

[Research](#research-contribution) · [Models](#models) · [Attacks](#attacks) · [Defenses](#defenses) · [Reproduce](#reproducing-results) · [Roadmap](#roadmap)

</div>

---

## Research Contribution

**Hypothesis:** Adversarial perturbations crafted against a multimodal model (CLIP-lite) to disrupt image-to-text retrieval will transfer to a standalone image classifier (ResNet-18), because both models share similar low-level feature representations.

**Method:** Optimize perturbation δ against CLIP-lite's contrastive loss → apply δ to standalone ResNet-18 → measure transfer rate vs. direct PGD attack (upper bound).

**Why this matters:** If multimodal attacks transfer to unimodal classifiers, deploying CLIP anywhere in a pipeline could compromise downstream models. This has real implications for robustness in production ML systems.

All models are implemented from scratch in PyTorch — no pretrained weights, no torchvision.models imports. Every attack is implemented directly from the original paper's equations.

## Models

All trained from scratch — no `torchvision.models` imports.

| Model | Dataset | Architecture | Target Accuracy |
|-------|---------|-------------|-----------------|
| ResNet-18 | CIFAR-10 | 4 residual blocks, batch norm, skip connections | ≥93% clean |
| Transformer | AG News | 6-layer encoder, 4 heads, 256-dim | ≥91% clean |
| CLIP-lite | Flickr8k | ResNet-18 image + 4-layer transformer text + contrastive loss | ≥60% R@1 |

## Attacks

All from the original paper's math — no `torchattacks` library.

| Attack | Paper | Method |
|--------|-------|--------|
| FGSM | Goodfellow et al., 2015 | `x_adv = x + ε · sign(∇_x L(θ, x, y))` |
| PGD-20 | Madry et al., 2018 | 20-step iterative with projection to ε-ball |
| TextFooler | Jin et al., 2020 | Word importance → synonym substitution → similarity filter |
| **CrossModal** | **Novel** | Image perturbation optimized against CLIP-lite, measured for transfer to ResNet-18 |

## Defenses

| Defense | Type | Method |
|---------|------|--------|
| Adversarial Training | Empirical | PGD-7 during training (Madry et al.) |
| Input Preprocessing | Empirical | JPEG compression + bit-depth reduction |
| Randomized Smoothing | **Certified** | Gaussian noise + Monte Carlo sampling (Cohen et al., 2019) |

## Project Structure

```
neurolens/
├── models/
│   ├── resnet.py              # ResNet-18 from scratch
│   ├── transformer.py         # 6-layer text classifier from scratch
│   └── dual_encoder.py        # CLIP-lite (image + text encoders)
├── attacks/
│   ├── base.py                # Abstract Attack interface
│   ├── fgsm.py                # x_adv = x + ε·sign(∇L)
│   ├── pgd.py                 # Iterative with ε-ball projection
│   ├── textfooler.py          # Word substitution attack
│   └── crossmodal_transfer.py # Novel: CLIP → ResNet transfer
├── defenses/
│   ├── adversarial_training.py    # PGD-AT training loop
│   ├── preprocessing.py           # JPEG + bit-depth transforms
│   └── randomized_smoothing.py    # Certified robustness
├── eval/
│   ├── attack_success_rate.py     # ASR vs epsilon curves
│   ├── transfer_matrix.py         # Cross-model transfer rates
│   └── figures.py                 # Publication-quality plots
├── configs/                   # YAML experiment configs
├── experiments/               # W&B sweep definitions
├── demo/app.py                # Streamlit interactive demo
├── paper/                     # LaTeX source (NeurIPS template)
├── tests/
└── notebooks/
```

## Reproducing Results

```bash
git clone https://github.com/Vyshnavi-d-p-3/neurolens.git
cd neurolens
pip install -r requirements.txt

# Train models
python -m models.resnet --config configs/models/resnet18_cifar10.yaml
python -m models.transformer --config configs/models/transformer_agnews.yaml

# Run attacks
python -m eval.attack_success_rate --config configs/attacks/pgd.yaml

# All experiments (requires GPU)
bash experiments/run_all.sh
```

All experiments are tracked in [Weights & Biases](https://wandb.ai/) with pinned random seeds for full reproducibility.

## Experiment Design

**Table 1:** Attack Success Rate vs. Perturbation Budget (ε)

**Table 2:** Defense Effectiveness at ε=0.1

**Table 3:** Cross-Modal Transfer Matrix — the key result

| Source Attack | Target: ResNet-18 | Target: Transformer | Target: CLIP-lite |
|--------------|-------------------|--------------------|--------------------|
| PGD on ResNet-18 | 100% (white-box) | ? (transfer) | ? (transfer) |
| PGD on Transformer | ? (transfer) | 100% (white-box) | ? (transfer) |
| **CrossModal on CLIP-lite** | **? (transfer)** | ? (transfer) | 100% (white-box) |

## Roadmap

- [x] Project structure + experiment design
- [x] ResNet-18 from scratch
- [x] Transformer classifier from scratch
- [x] FGSM attack implementation
- [x] PGD-20 attack implementation
- [x] Randomized smoothing defense
- [ ] CLIP-lite dual encoder
- [ ] Cross-modal transfer attack
- [ ] TextFooler implementation
- [ ] Adversarial training defense
- [ ] Full experiment sweep
- [ ] Paper writing (NeurIPS workshop template)
- [ ] arXiv submission

## License

MIT

---

<div align="center">
<sub>Built by <a href="https://github.com/Vyshnavi-d-p-3">Vyshnavi D P</a></sub>
</div>
