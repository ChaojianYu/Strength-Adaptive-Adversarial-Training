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


## Contact
For questions, please contact: chaojianyu@hust.edu.cn
