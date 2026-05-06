# Artifact Guided EfficientViT: A Robust Hybrid CNN and Transformer Framework for AI Generated Image Detection

<p align="center">
  <b>AG-EfficientViT V3</b><br>
  Fine-tuned EfficientNetB0 and ViT-Tiny branches with artifact-sensitive logit fusion for real vs AI-generated image detection.<br>
  <sub>98.865% Accuracy · 98.864% F1-score · 0.999116 AUC on CIFAKE</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Task-AI%20Generated%20Image%20Detection-blue" />
  <img src="https://img.shields.io/badge/Dataset-CIFAKE-green" />
  <img src="https://img.shields.io/badge/Framework-PyTorch-red" />
  <img src="https://img.shields.io/badge/Final%20Model-AG--EfficientViT%20V3-purple" />
</p>

---

## Abstract

The rapid advancement of generative artificial intelligence has increased the difficulty of distinguishing authentic images from AI-generated content. Existing convolutional neural networks and Vision Transformer models have shown strong performance in visual classification tasks; however, single-branch architectures may not fully capture both local artifact traces and global semantic inconsistencies in synthetic images. This repository presents **Artifact Guided EfficientViT (AG-EfficientViT)**, a hybrid CNN-Transformer framework for AI-generated image detection. The final proposed model, **AG-EfficientViT V3**, integrates a fine-tuned EfficientNetB0 branch, a fine-tuned ViT-Tiny branch, and an artifact-oriented branch using calibrated logit-level fusion. On the CIFAKE benchmark, AG-EfficientViT V3 achieves **98.865% accuracy**, **98.864% F1-score**, and **0.999116 AUC**, outperforming the evaluated CNN baseline, Transformer baseline, simple hybrid baseline, and earlier AG-EfficientViT variants.

---

## Table of Contents

- [1. Introduction](#1-introduction)
  - [1.1 Background](#11-background)
  - [1.2 Research Motivation](#12-research-motivation)
  - [1.3 Main Contributions](#13-main-contributions)
- [2. Proposed Method](#2-proposed-method)
  - [2.1 Problem Formulation](#21-problem-formulation)
  - [2.2 Overall Architecture](#22-overall-architecture)
  - [2.3 Branch-Level Representation Learning](#23-branch-level-representation-learning)
  - [2.4 Artifact-Guided Logit-Level Fusion](#24-artifact-guided-logit-level-fusion)
  - [2.5 Training Objective](#25-training-objective)
  - [2.6 Evaluation Metrics](#26-evaluation-metrics)
  - [2.7 Difference Between V1, V2, and V3](#27-difference-between-v1-v2-and-v3)
- [3. Dataset and Experimental Setup](#3-dataset-and-experimental-setup)
  - [3.1 Dataset Description](#31-dataset-description)
  - [3.2 Dataset Organization](#32-dataset-organization)
  - [3.3 Experimental Protocol](#33-experimental-protocol)
- [4. Experimental Results and Analysis](#4-experimental-results-and-analysis)
  - [4.1 Clean-Test Performance](#41-clean-test-performance)
  - [4.2 Ablation Study](#42-ablation-study)
  - [4.3 Robustness Analysis](#43-robustness-analysis)
  - [4.4 Qualitative Results](#44-qualitative-results)
  - [4.5 Visual Interpretability](#45-visual-interpretability)
  - [4.6 Comparison with Related Studies](#46-comparison-with-related-studies)
- [5. Repository Usage](#5-repository-usage)
  - [5.1 Repository Structure](#51-repository-structure)
  - [5.2 Installation](#52-installation)
  - [5.3 Training](#53-training)
  - [5.4 Evaluation](#54-evaluation)
- [6. Limitations](#6-limitations)
- [7. Future Work](#7-future-work)
- [8. Citation](#8-citation)
- [9. Author and Contact](#9-author-and-contact)

---

# 1. Introduction

## 1.1 Background

AI-generated images are becoming increasingly realistic due to the rapid development of generative models. This creates a growing need for reliable detection systems that can distinguish real images from synthetic images in digital media, security, education, and forensic applications.

Convolutional neural networks are effective in capturing local spatial and texture patterns, while Vision Transformers are strong in modeling global dependencies and semantic-level inconsistencies. However, AI-generated image detection often requires both types of information: local artifact traces and global representation cues.

## 1.2 Research Motivation

A simple combination of CNN and Transformer features does not always guarantee improved detection performance. In this project, the simple EfficientViT-Hybrid baseline does not outperform the strongest single-branch ViT-Tiny baseline. This motivates a more careful fusion strategy that uses fine-tuned branch initialization and artifact-guided decision fusion.

The main research motivation is therefore to develop a hybrid detection model that preserves the strengths of CNN and Transformer branches while introducing artifact-aware correction at the decision level.

## 1.3 Main Contributions

The main contributions of this repository are summarized as follows:

1. **Hybrid CNN-Transformer detection framework** for AI-generated image detection.
2. **Fine-tuned branch initialization** using separately trained EfficientNetB0 and ViT-Tiny checkpoints.
3. **Artifact-guided logit-level fusion** that combines CNN logits, Transformer logits, and artifact logits.
4. **Transparent variant-level ablation** covering EfficientNetB0, ViT-Tiny, EfficientViT-Hybrid, AG-EfficientViT V1, V2, and V3.
5. **Robustness, qualitative, and interpretability-oriented analysis** to support journal-style reporting.

---

# 2. Proposed Method

## 2.1 Problem Formulation

Let `x` denote an input RGB image and `y` denote its binary ground-truth label. The model is trained to classify an image as either AI-generated or real.

**Equation (1). Input and label definition**

```math
x \in \mathbb{R}^{H \times W \times 3}, \qquad y \in \{0,1\}
```

where `0` denotes an AI-generated image and `1` denotes a real image, following the binary class indexing used in this repository.

The model learns a function that maps an input image to a two-class prediction.

**Equation (2). Binary image classification function**

```math
f_{\theta}(x) \rightarrow \hat{y}
```

The network produces a two-dimensional logit vector.

**Equation (3). Two-class logit vector**

```math
z = [z_0, z_1] \in \mathbb{R}^{2}
```

The posterior probability for class `k` is computed using the softmax function.

**Equation (4). Softmax probability**

```math
p(y=k \mid x) =
\frac{\exp(z_k)}
{\sum_{j=0}^{1}\exp(z_j)},
\qquad k \in \{0,1\}
```

The final predicted class is obtained by selecting the class with the maximum posterior probability.

**Equation (5). Final prediction**

```math
\hat{y}
=
\arg\max_{k \in \{0,1\}}
p(y=k \mid x)
```

## 2.2 Overall Architecture

The proposed **AG-EfficientViT V3** framework consists of three parallel branches and a final logit-level fusion stage.

<p align="center">
  <img src="docs/figures/ag_efficientvit_v3_architecture.png" alt="AG-EfficientViT V3 Architecture" width="900"/>
</p>

**Figure 1.** Overall architecture of the proposed AG-EfficientViT V3 framework. The model combines a fine-tuned EfficientNetB0 branch, a fine-tuned ViT-Tiny branch, and an artifact branch through logit-level fusion.

The model uses three complementary decision sources:

1. an EfficientNetB0 branch for local spatial and texture cues,
2. a ViT-Tiny branch for global semantic representation, and
3. an artifact branch for synthetic-trace-oriented evidence.

## 2.3 Branch-Level Representation Learning

The CNN branch produces a class-logit vector from the input image.

**Equation (6). EfficientNetB0 branch logits**

```math
z_{\mathrm{cnn}}
=
f_{\mathrm{cnn}}(x;\theta_{\mathrm{cnn}}^{*})
```

where `theta_cnn*` denotes the fine-tuned EfficientNetB0 parameters.

The Transformer branch produces a second class-logit vector.

**Equation (7). ViT-Tiny branch logits**

```math
z_{\mathrm{vit}}
=
f_{\mathrm{vit}}(x;\theta_{\mathrm{vit}}^{*})
```

where `theta_vit*` denotes the fine-tuned ViT-Tiny parameters.

The artifact branch extracts artifact-sensitive features and maps them into class logits.

**Equation (8). Artifact feature extraction**

```math
h_{\mathrm{art}}
=
g_{\mathrm{art}}(x;\theta_{\mathrm{art}})
```

**Equation (9). Artifact branch logits**

```math
z_{\mathrm{art}}
=
W_{\mathrm{art}} h_{\mathrm{art}} + b_{\mathrm{art}}
```

The artifact branch is not intended to replace the CNN or Transformer branch. Instead, it provides an additional artifact-sensitive decision signal that can support the final classification.

## 2.4 Artifact-Guided Logit-Level Fusion

AG-EfficientViT V3 performs fusion at the logit level rather than through naïve feature concatenation. The three branch logits are combined into the final logit vector.

**Equation (10). Logit-level fusion**

```math
z_{\mathrm{final}}
=
\alpha_{\mathrm{cnn}} z_{\mathrm{cnn}}
+
\alpha_{\mathrm{vit}} z_{\mathrm{vit}}
+
\alpha_{\mathrm{art}} z_{\mathrm{art}}
```

The fusion weights are normalized using the softmax function.

**Equation (11). Fusion-weight normalization**

```math
[\alpha_{\mathrm{cnn}},\alpha_{\mathrm{vit}},\alpha_{\mathrm{art}}]
=
\mathrm{softmax}(w)
```

where `w` is a learnable vector of three fusion logits.

In the implemented V3 configuration, the initial fusion vector is:

**Equation (12). Initial fusion logits**

```math
w = [1.0,\;2.5,\;-2.0]
```

This gives the following initial normalized fusion weights:

**Equation (13). Initial normalized fusion weights**

```math
\alpha_{\mathrm{cnn}} \approx 0.1808,
\qquad
\alpha_{\mathrm{vit}} \approx 0.8102,
\qquad
\alpha_{\mathrm{art}} \approx 0.0090
```

This initialization is intentionally ViT-dominant, CNN-supportive, and artifact-conservative. It preserves the strong decision boundary of the fine-tuned ViT-Tiny branch while still allowing CNN and artifact information to contribute.

The final class probability is computed from the fused logits.

**Equation (14). Final probability from fused logits**

```math
p(y=k \mid x)
=
\mathrm{softmax}(z_{\mathrm{final}})_k
```

## 2.5 Training Objective

The model is optimized using cross-entropy loss over the binary training set.

**Equation (15). Training set**

```math
\mathcal{D}
=
\{(x_i,y_i)\}_{i=1}^{N}
```

**Equation (16). Cross-entropy objective**

```math
\mathcal{L}_{\mathrm{CE}}
=
-\frac{1}{N}
\sum_{i=1}^{N}
\sum_{k=0}^{1}
\mathbf{1}(y_i=k)
\log p(y=k \mid x_i)
```

During V3 training, the fine-tuned CNN and ViT branches provide strong initialized decision signals, while the artifact branch and fusion mechanism refine the final prediction.

## 2.6 Evaluation Metrics

The evaluation uses accuracy, precision, recall, F1-score, and AUC.

**Equation (17). Accuracy**

```math
\mathrm{Accuracy}
=
\frac{TP+TN}
{TP+TN+FP+FN}
```

**Equation (18). Precision**

```math
\mathrm{Precision}
=
\frac{TP}
{TP+FP}
```

**Equation (19). Recall**

```math
\mathrm{Recall}
=
\frac{TP}
{TP+FN}
```

**Equation (20). F1-score**

```math
\mathrm{F1}
=
2 \cdot
\frac{\mathrm{Precision}\cdot\mathrm{Recall}}
{\mathrm{Precision}+\mathrm{Recall}}
```

AUC measures the area under the ROC curve and reflects the ranking quality of the predicted probabilities.

## 2.7 Difference Between V1, V2, and V3

The AG-EfficientViT variants are retained to make the model development process transparent. V1 and V2 are not presented as final models; they are reported as ablation variants that explain why V3 was selected.

| Variant | Main Purpose | Branch Initialization | Fusion Design | Main Observation |
|---|---|---|---|---|
| **AG-EfficientViT V1** | Initial artifact-guided hybrid design | Generic pretrained initialization | Artifact-guided weighted fusion | Competitive, but below ViT-Tiny and V3 |
| **AG-EfficientViT V2** | Revised interaction/gating variant | Generic pretrained initialization | Gated concatenation / interaction fusion | Higher recall, but reduced precision and accuracy |
| **AG-EfficientViT V3** | Final proposed model | Fine-tuned EfficientNetB0 + fine-tuned ViT-Tiny checkpoints | Calibrated logit-level fusion | Best overall clean CIFAKE performance |

### 2.7.1 AG-EfficientViT V1

AG-EfficientViT V1 introduced the initial artifact-guided hybrid idea. It combined CNN, Transformer, and artifact-oriented information, but the branch initialization and fusion design were not yet optimal.

**Equation (21). V1 result summary**

```math
\mathrm{Accuracy}_{\mathrm{V1}} = 98.670\%,
\qquad
\mathrm{F1}_{\mathrm{V1}} = 98.668\%,
\qquad
\mathrm{AUC}_{\mathrm{V1}} = 0.998713
```

This result shows that adding an artifact branch alone is not sufficient to outperform the strongest single-branch baseline.

### 2.7.2 AG-EfficientViT V2

AG-EfficientViT V2 explored a revised gated fusion strategy. This variant achieved high recall but lower precision, suggesting a stronger tendency toward one class.

**Equation (22). V2 result summary**

```math
\mathrm{Accuracy}_{\mathrm{V2}} = 98.625\%,
\qquad
\mathrm{Precision}_{\mathrm{V2}} = 98.225\%,
\qquad
\mathrm{Recall}_{\mathrm{V2}} = 99.040\%,
\qquad
\mathrm{F1}_{\mathrm{V2}} = 98.631\%,
\qquad
\mathrm{AUC}_{\mathrm{V2}} = 0.998733
```

This behavior indicates that a stronger gated interaction does not necessarily improve calibration or overall accuracy.

### 2.7.3 AG-EfficientViT V3

AG-EfficientViT V3 is the final proposed model. Its key difference is not simply the presence of three branches, but the use of fine-tuned branch initialization and calibrated logit-level fusion.

**Equation (23). V3 result summary**

```math
\mathrm{Accuracy}_{\mathrm{V3}} = 98.865\%,
\qquad
\mathrm{Precision}_{\mathrm{V3}} = 98.938\%,
\qquad
\mathrm{Recall}_{\mathrm{V3}} = 98.790\%,
\qquad
\mathrm{F1}_{\mathrm{V3}} = 98.864\%,
\qquad
\mathrm{AUC}_{\mathrm{V3}} = 0.999116
```

Thus, V3 demonstrates that the best strategy is not merely to stack modules, but to use strong branch-level initialization and a stable decision-level fusion mechanism.

---

# 3. Dataset and Experimental Setup

## 3.1 Dataset Description

This project uses the **CIFAKE** dataset for binary classification of real and AI-generated images.

| Split | Real Images | Fake Images | Total |
|---|---:|---:|---:|
| Train | 50,000 | 50,000 | 100,000 |
| Test | 10,000 | 10,000 | 20,000 |
| Total | 60,000 | 60,000 | 120,000 |

The dataset is not included in this repository and should be downloaded separately.

## 3.2 Dataset Organization

The dataset should be organized as follows:

```text
data/
└── CIFAKE/
    ├── train/
    │   ├── fake/
    │   └── real/
    └── test/
        ├── fake/
        └── real/
```

## 3.3 Experimental Protocol

All models are evaluated on the CIFAKE test set using the same binary classification protocol. The final reported comparison includes:

- EfficientNetB0 baseline,
- ViT-Tiny baseline,
- EfficientViT-Hybrid baseline,
- AG-EfficientViT V1,
- AG-EfficientViT V2, and
- AG-EfficientViT V3.

---

# 4. Experimental Results and Analysis

## 4.1 Clean-Test Performance

The clean-test benchmark on CIFAKE is shown in Table 1.

| Rank | Model | Best Epoch | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) | AUC |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | **AG-EfficientViT V3** | 10 | **98.865** | **98.938** | 98.790 | **98.864** | **0.999116** |
| 2 | ViT-Tiny | 20 | 98.750 | 98.809 | 98.690 | 98.749 | 0.998620 |
| 3 | EfficientViT-Hybrid | 15 | 98.710 | 98.778 | 98.640 | 98.709 | 0.998759 |
| 4 | AG-EfficientViT V1 | 13 | 98.670 | 98.807 | 98.530 | 98.668 | 0.998713 |
| 5 | AG-EfficientViT V2 | 18 | 98.625 | 98.225 | **99.040** | 98.631 | 0.998733 |
| 6 | EfficientNetB0 | 19 | 98.065 | 98.214 | 97.910 | 98.062 | 0.997674 |

**Table 1.** Clean-test performance comparison on CIFAKE.

<p align="center">
  <img src="docs/figures/main_results_comparison.png" alt="Main Results Comparison" width="900"/>
</p>

**Figure 2.** Performance comparison of all evaluated models on the CIFAKE clean test set.

The results show that **AG-EfficientViT V3** achieves the best overall clean-test performance. Compared with ViT-Tiny, the strongest single-branch baseline, AG-EfficientViT V3 improves accuracy from **98.750%** to **98.865%**, F1-score from **98.749%** to **98.864%**, and AUC from **0.998620** to **0.999116**.

## 4.2 Ablation Study

The ablation study evaluates the contribution of each design choice, including single-branch baselines, simple hybridization, artifact-guided variants, and the final AG-EfficientViT V3 model.

| Variant | CNN Branch | ViT Branch | Artifact Branch | Fusion Strategy | Accuracy (%) | F1-score (%) | AUC |
|---|:---:|:---:|:---:|---|---:|---:|---:|
| EfficientNetB0 | ✓ | - | - | - | 98.065 | 98.062 | 0.997674 |
| ViT-Tiny | - | ✓ | - | - | 98.750 | 98.749 | 0.998620 |
| EfficientViT-Hybrid | ✓ | ✓ | - | Feature concatenation | 98.710 | 98.709 | 0.998759 |
| AG-EfficientViT V1 | ✓ | ✓ | ✓ | Weighted / artifact-guided fusion | 98.670 | 98.668 | 0.998713 |
| AG-EfficientViT V2 | ✓ | ✓ | ✓ | Gated concatenation / interaction fusion | 98.625 | 98.631 | 0.998733 |
| **AG-EfficientViT V3** | ✓ | ✓ | ✓ | **Logit-level fusion with fine-tuned branch initialization** | **98.865** | **98.864** | **0.999116** |

**Table 2.** Ablation study of AG-EfficientViT variants and baseline models.

<p align="center">
  <img src="docs/figures/ablation_results.png" alt="Ablation Results" width="900"/>
</p>

**Figure 3.** Ablation visualization across baselines and AG-EfficientViT variants.

The ablation results show that **naïve hybridization is not sufficient**. Although EfficientViT-Hybrid combines CNN and Transformer features, it does not surpass ViT-Tiny. The strongest result is achieved by AG-EfficientViT V3, indicating that fine-tuned branch initialization and logit-level fusion are more effective than simple concatenation or earlier artifact-guided fusion variants.

## 4.3 Robustness Analysis

Robustness evaluation was performed under multiple degradation conditions, including JPEG compression, Gaussian blur, resize degradation, and additive noise.

<p align="center">
  <img src="docs/figures/robustness_plot.png" alt="Robustness Plot" width="950"/>
</p>

**Figure 4.** Robustness comparison of key models under image degradation.

The robustness analysis shows that AG-EfficientViT V3 performs strongly under clean and JPEG-compressed conditions. However, severe blur, resize degradation, and additive noise remain challenging. Therefore, the robustness claim should be interpreted as degradation-specific rather than universal robustness across all corruptions.

## 4.4 Qualitative Results

Qualitative results are used to inspect representative predictions, including correct classifications, hard cases, and failure cases.

<p align="center">
  <img src="docs/figures/qualitative_results.png" alt="Qualitative Results" width="950"/>
</p>

**Figure 5.** Representative qualitative examples on CIFAKE.

| Case ID | Image Type | Ground Truth | ViT-Tiny | EfficientViT-Hybrid | AG-EfficientViT V3 | Notes |
|---|---|---|---|---|---|---|
| Q1 | Real | Real | Correct | Correct | Correct | Easy real sample |
| Q2 | Fake | Fake | Correct | Correct | Correct | Easy generated sample |
| Q3 | Fake | Fake | Incorrect | Incorrect | Correct | V3 improves prediction |
| Q4 | Real | Real | Correct | Incorrect | Correct | Artifact-guided correction helps |
| Q5 | Real/Fake | Real/Fake | Correct | Correct | Incorrect | Failure case |
| Q6 | Real/Fake | Real/Fake | Incorrect | Correct | Incorrect | Ambiguous case |

**Table 3.** Template for qualitative case analysis.

The qualitative analysis should emphasize where the final model improves over baselines and where it still fails. This is important for journal-style reporting because it shows not only numerical performance, but also model behavior under visually meaningful cases.

## 4.5 Visual Interpretability

Visual interpretability can be used to analyze which image regions influence the prediction of AG-EfficientViT V3.

<p align="center">
  <img src="docs/figures/gradcam_examples.png" alt="Grad-CAM Visualization" width="950"/>
</p>

**Figure 6.** Grad-CAM examples for AG-EfficientViT V3.

The interpretability analysis should examine whether the model focuses on artifact-relevant regions such as unnatural textures, edge inconsistencies, object boundaries, or background irregularities. Failure cases should also be inspected to identify misleading activation patterns.

---


## 4.6 Comparison with Related Studies

To position the proposed method against recent AI-generated image detection studies, Table 5 summarizes publicly reported results from related works that evaluate real-versus-AI-generated image classification, particularly on CIFAKE or closely related synthetic image detection benchmarks.

> **Important note.** The comparison is intended as a literature-level positioning table, not as a fully controlled head-to-head benchmark. Reported values may be affected by differences in image resolution, preprocessing, train-test protocol, data augmentation, backbone capacity, random seeds, thresholding strategy, and whether AUC refers to ROC-AUC or PR-AUC.

| Study | Year | Dataset / Task | Model / Method | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) | AUC |
|---|---:|---|---|---:|---:|---:|---:|---:|
| Bird and Lotfi [R1] | 2024 | CIFAKE | CNN with explainable Grad-CAM analysis | 92.98 | N/R | N/R | N/R | N/R |
| Wang et al. [R2] | 2024 | CIFAKE | Transfer learning with DenseNet | 97.74 | N/R | N/R | N/R | N/R |
| Islam et al. [R3] | 2024 | CIFAKE | MEXFIC meta-ensemble classifier | 94.00 | N/R | N/R | N/R | N/R |
| Gunukula et al. [R4] | 2025 | CIFAKE | Hybrid SE-ResNet50 attention model | 96.12 | 97.04 | 88.94 | 92.82 | 0.9862 |
| EfficientNet loss-variant study [R5] | 2025/2026 | CIFAKE | EfficientNet-B3 with attention-enhanced CE | 97.58 | 96.56 | 98.67 | 97.61 | 0.9973* |
| **This repository** | **2026** | **CIFAKE** | **AG-EfficientViT V3** | **98.865** | **98.938** | **98.790** | **98.864** | **0.999116** |

\* Reported as PR-AUC in the original study.

The comparison shows that **AG-EfficientViT V3 achieves the strongest accuracy, F1-score, and AUC among the selected recent CIFAKE-oriented studies listed in Table 5**. Compared with conventional CNN-based detectors, the proposed model benefits from complementary feature modeling: EfficientNetB0 captures local convolutional artifact cues, ViT-Tiny contributes global contextual reasoning, and the artifact-guided branch provides additional evidence for subtle generative traces. The final V3 design further improves reliability by initializing the CNN and Transformer branches from their separately fine-tuned checkpoints before performing calibrated logit-level fusion.

This result should be interpreted carefully. The current repository demonstrates strong performance on CIFAKE, but broader claims such as universal state-of-the-art performance require external dataset validation, multi-generator evaluation, and controlled reimplementation of competing methods under the same training and testing protocol.

### References for Table 5

[R1] J. J. Bird and A. Lotfi, “CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images,” *IEEE Access*, 2024.

[R2] Y. Wang, Y. Hao, and A. X. Cong, “Harnessing Machine Learning for Discerning AI-Generated Synthetic Images,” arXiv, 2024.

[R3] M. T. Islam, I. H. Lee, A. I. Alzahrani, and K. Muhammad, “MEXFIC: A Meta Ensemble eXplainable Approach for AI-Synthesized Fake Image Classification,” *Alexandria Engineering Journal*, 2024.

[R4] A. R. Gunukula, H. Das Gupta, and V. S. Sheng, “Detecting AI-Generated Images Using a Hybrid ResNet-SE Attention Model,” *Applied Sciences*, 2025.

[R5] F. Bayram, “Comparison of Deep Learning Approaches for Fake Image Classification,” *International Journal of Advanced Natural Sciences and Engineering Researches*, 2026.

---

# 5. Repository Usage

## 5.1 Repository Structure

```text
AG-EfficientViT/
│
├── configs/
│   └── cifake.yaml
│
├── datasets/
│   ├── __init__.py
│   └── cifake_dataset.py
│
├── docs/
│   └── figures/
│       ├── ag_efficientvit_v3_architecture.png
│       ├── main_results_comparison.png
│       ├── ablation_results.png
│       ├── robustness_plot.png
│       ├── qualitative_results.png
│       └── gradcam_examples.png
│
├── models/
│   ├── __init__.py
│   ├── efficientnet_baseline.py
│   ├── vit_baseline.py
│   ├── efficientvit_baseline.py
│   ├── artifact_branch.py
│   ├── fusion.py
│   ├── fusion_v2.py
│   ├── ag_efficientvit.py
│   ├── ag_efficientvit_v2.py
│   └── ag_efficientvit_v3.py
│
├── results/
│   ├── tables/
│   ├── figures/
│   └── logs/
│
├── scripts/
│   ├── generate_repo_figures.py
│   ├── generate_qualitative_results.py
│   └── generate_gradcam_examples.py
│
├── train.py
├── train_vit.py
├── train_efficientvit.py
├── train_ag_efficientvit.py
├── train_ag_efficientvit_v2.py
├── train_ag_efficientvit_v3.py
│
├── evaluate.py
├── evaluate_vit.py
├── evaluate_efficientvit.py
├── evaluate_ag_efficientvit.py
├── evaluate_ag_efficientvit_v2.py
├── evaluate_ag_efficientvit_v3.py
│
├── robustness_test.py
├── summarize_results.py
├── FINAL_PROJECT_SUMMARY.md
├── requirements.txt
└── README.md
```

## 5.2 Installation

Create and activate a Python environment:

```bash
conda create -n ag_efficientvit python=3.11 -y
conda activate ag_efficientvit
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 5.3 Training

Train EfficientNetB0 baseline:

```bash
python train.py --epochs 20 --batch-size 128
```

Train ViT-Tiny baseline:

```bash
python train_vit.py --epochs 20 --batch-size 128
```

Train EfficientViT-Hybrid baseline:

```bash
python train_efficientvit.py --epochs 20 --batch-size 64
```

Train AG-EfficientViT V3:

```bash
python train_ag_efficientvit_v3.py --epochs 10 --batch-size 128 --lr 0.0001 --weight-decay 0.0001
```

## 5.4 Evaluation

Evaluate AG-EfficientViT V3:

```bash
python evaluate_ag_efficientvit_v3.py --checkpoint checkpoints/ag_efficientvit_v3_cifake_best.pth --batch-size 128
```

Summarize all model results:

```bash
python summarize_results.py
```

Run robustness evaluation:

```bash
python robustness_test.py --models key --batch-size 128
```

Generate repository figures:

```bash
python scripts/generate_repo_figures.py
python scripts/generate_qualitative_results.py
python scripts/generate_gradcam_examples.py
```

---

# 6. Limitations

This repository currently focuses on the CIFAKE dataset. Although AG-EfficientViT V3 achieves strong clean and JPEG-compressed performance, several limitations remain:

1. External dataset validation is still required to support stronger generalization claims.
2. Severe blur, resize degradation, and additive noise remain challenging.
3. Robustness claims should be interpreted carefully and degradation-specifically.
4. More qualitative and interpretability analysis is needed for stronger scientific evidence.
5. Deployment efficiency and inference latency should be evaluated in future experiments.

---

# 7. Future Work

Future work may include:

1. Cross-dataset validation on external AI-generated image datasets.
2. More stable robustness-aware training strategies.
3. Grad-CAM and attention-based interpretability comparison across baselines.
4. Calibration analysis and threshold optimization.
5. Lightweight deployment and inference benchmarking.
6. Extension to multi-generator and open-set AI-generated image detection.

---

# 8. Citation

If you find this repository useful, please cite it as:

```bibtex
@misc{ag_efficientvit_2026,
  title  = {Artifact Guided EfficientViT: A Robust Hybrid CNN and Transformer Framework for AI Generated Image Detection},
  author = {Mochamad Rizal Fauzan},
  year   = {2026},
  note   = {Research implementation for AI-generated image detection}
}
```

---

# 9. Author and Contact

**Mochamad Rizal Fauzan**  
AI Engineer | Computer Vision Researcher | Embedded Systems and LLM Researcher

For academic discussion, collaboration, or repository-related questions, please open an issue or contact the repository author through GitHub.
