# Strength-Adaptive Adversarial Training

This repository provides the official PyTorch implementation of **Strength-Adaptive Adversarial Training (SAAT)** proposed in:

> **Chaojian Yu, Dawei Zhou, Li Shen, Jun Yu, Bo Han, Mingming Gong, Nannan Wang, and Tongliang Liu.**  
> **Strength-Adaptive Adversarial Training.**  
> *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2026.*

**Paper:** [IEEE TPAMI](https://doi.org/10.1109/TPAMI.2026.3684741)

**Authors:** Chaojian Yu, Dawei Zhou, Li Shen, Jun Yu, Bo Han, Mingming Gong, Nannan Wang, and Tongliang Liu

---

## Overview

Adversarial Training (AT) is one of the most effective approaches for improving the adversarial robustness of deep neural networks. However, conventional AT relies on a **fixed, pre-specified perturbation budget**, which can lead to two important issues:

1. **Robustness disparity:** the same perturbation budget may result in substantially different trade-offs between natural accuracy and adversarial robustness for networks with different capacities.
2. **Robust overfitting:** as the model becomes increasingly robust during training, adversarial examples generated under a fixed perturbation budget may become progressively weaker, creating an imbalance between the model and the adversary.

To address these issues, we propose **Strength-Adaptive Adversarial Training (SAAT)**. Instead of constraining adversarial examples with a fixed perturbation budget, SAAT imposes a **minimum adversarial loss constraint** and adaptively adjusts the perturbation budget to generate adversarial training examples with the desired attack strength.

## Repository Structure

The repository contains four implementations:

```text
.
├── AT/
│   └── main.py
│   └── ...
│
├── AT-AWP/
│   └── main.py
│   └── ...
│
├── SAAT/
│   └── main.py
│   └── ...
│
├── SAAT-AWP/
│   └── main.py
│   └── ...
│
└── README.md
```

The four implementations correspond to:

| Method | Description |
|:---|:---|
| **AT** | Standard adversarial training with PGD |
| **AT-AWP** | Standard adversarial training combined with Adversarial Weight Perturbation (AWP) |
| **SAAT** | Proposed Strength-Adaptive Adversarial Training |
| **SAAT-AWP** | SAAT combined with Adversarial Weight Perturbation |

---

## Training

### Standard Adversarial Training (AT)

To train a standard PGD-based adversarially trained model:

```bash
python main.py \
    --model PreActResNet18 \
    --batch-size 128 \
    --epochs 200 \
    --lr-max 0.1 \
    --attack pgd \
    --norm l_inf
```

### Strength-Adaptive Adversarial Training (SAAT)

The proposed SAAT implementation uses SA-PGD to adaptively determine the perturbation strength.

```bash
python main.py \
    --model PreActResNet18 \
    --batch-size 128 \
    --epochs 200 \
    --lr-max 0.1 \
    --attack pgd \
    --norm l_inf
```

The main SAAT-specific parameters are implemented directly in `sa_pgd()`:

```text
minimum adversarial loss ρ = 1.7
maximum perturbation budget = 14/255
budget step size τ = 2/255
SA-PGD steps K = 3
```

## Evaluation

The code supports evaluation under natural and adversarial inputs:

- **Natural accuracy**
- **PGD-20 accuracy**
- **AutoAttack (AA)**

AutoAttack includes APGD-CE, APGD-DLR, FAB, and Square Attack and is used as a comprehensive robustness evaluation protocol. 

## Why SAAT?

Compared with conventional AT, SAAT provides three main advantages:

- **Adaptive attack strength:** the perturbation budget is dynamically adjusted according to the current robustness of the model.
- **Reduced robust overfitting:** stronger adversarial examples can be generated as the model becomes more robust.
- **Controllable robustness disparity:** the adversarial loss constraint $\rho$ provides a mechanism for controlling the allocation of model capacity between natural accuracy and adversarial robustness.

The paper demonstrates that SAAT consistently improves adversarial robustness over standard AT and can provide complementary gains when combined with existing methods such as AWP. 

---

## Citation

If you find this repository useful for your research, please consider citing:

```bibtex
@article{yu2026strength,
  title={Strength-Adaptive Adversarial Training},
  author={Yu, Chaojian and Zhou, Dawei and Shen, Li and Yu, Jun and Han, Bo and Gong, Mingming and Wang, Nannan and Liu, Tongliang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2026},
  publisher={IEEE}
}
```

---

## Acknowledgement

This implementation is built upon the PyTorch framework and draws on publicly available implementations of adversarial training and adversarial weight perturbation.
We thank the authors of the corresponding prior works for making their code publicly available.

[1] AT: https://github.com/locuslab/robust_overfitting

[2] AWP: https://github.com/csdongxian/AWP

[3] AutoAttack: https://github.com/fra31/auto-attack
