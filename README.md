# Artifact Guided EfficientViT: A Robust Hybrid CNN and Transformer Framework for AI Generated Image Detection

<p align="center">
  <b>AG-EfficientViT V3</b><br>
  A hybrid CNN-Transformer detection framework with artifact-guided logit fusion for real vs AI-generated image classification.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Task-AI%20Generated%20Image%20Detection-blue" />
  <img src="https://img.shields.io/badge/Dataset-CIFAKE-green" />
  <img src="https://img.shields.io/badge/Framework-PyTorch-red" />
  <img src="https://img.shields.io/badge/Final%20Model-AG--EfficientViT%20V3-purple" />
  <img src="https://img.shields.io/badge/Accuracy-98.865%25-brightgreen" />
  <img src="https://img.shields.io/badge/AUC-0.999116-orange" />
</p>

---

## Overview

The rapid development of generative artificial intelligence has made it increasingly difficult to distinguish authentic images from AI-generated content. Although convolutional neural networks (CNNs) and Vision Transformers (ViTs) have both demonstrated strong visual recognition capability, single-branch models may not fully capture the combination of **local artifact traces** and **global semantic inconsistencies** commonly found in synthetic imagery.

This repository presents **Artifact Guided EfficientViT**, a hybrid CNN-Transformer framework for AI-generated image detection. The final proposed model, **AG-EfficientViT V3**, combines:

- a **fine-tuned EfficientNetB0 branch** for local spatial and texture cues,
- a **fine-tuned ViT-Tiny branch** for global semantic representation,
- an **artifact branch** for synthetic-trace extraction, and
- a **logit-level fusion mechanism** for final decision making.

On the **CIFAKE** dataset, AG-EfficientViT V3 achieves the best overall clean-test performance among all evaluated models, reaching:

- **Accuracy:** 98.865%
- **Precision:** 98.938%
- **Recall:** 98.790%
- **F1-score:** 98.864%
- **AUC:** 0.999116

---

## Highlights

- Proposes **AG-EfficientViT V3**, a hybrid CNN-Transformer framework for AI-generated image detection.
- Combines **EfficientNetB0**, **ViT-Tiny**, and an **artifact branch** in a unified framework.
- Uses **fine-tuned branch initialization** from the best-performing CNN and Transformer baselines.
- Demonstrates that **simple CNN-Transformer concatenation is not sufficient** to guarantee the best performance.
- Achieves the **best clean CIFAKE result** among all evaluated baselines and ablation variants.
- Includes **baseline comparison**, **ablation study**, **robustness analysis**, and **qualitative evaluation**.

---

## Figure 1. Proposed Architecture

> Replace the placeholder below with your final architecture figure.

<p align="center">
  <img src="docs/figures/ag_efficientvit_v3_architecture.png" alt="AG-EfficientViT V3 Architecture" width="900"/>
</p>

**Figure 1.** Overview of the proposed **AG-EfficientViT V3** framework. The model combines a fine-tuned **EfficientNetB0 branch**, a fine-tuned **ViT-Tiny branch**, and an **artifact branch**. The outputs are integrated through **logit-level fusion** for final real/fake classification.

---

## Method Overview

The final proposed model, **AG-EfficientViT V3**, consists of the following components:

### 1. EfficientNetB0 Branch
This branch captures **local texture patterns**, **high-frequency structures**, and **fine-grained spatial cues** that are useful for detecting synthetic artifacts.

### 2. ViT-Tiny Branch
This branch captures **global visual context**, **long-range dependencies**, and **semantic consistency**, which are important for identifying broader structural irregularities.

### 3. Artifact Branch
The artifact branch focuses on **artifact-oriented features**, aiming to capture synthetic traces that may not be fully represented by the main CNN or Transformer branches.

### 4. Logit-Level Fusion
Instead of simple feature concatenation, AG-EfficientViT V3 performs **logit-level fusion** over:
- the fine-tuned EfficientNetB0 logits,
- the fine-tuned ViT-Tiny logits, and
- the artifact branch logits.

This design preserves the strengths of each branch while enabling artifact-guided correction in the final decision stage.

---

## Dataset

This project uses the **CIFAKE** dataset for binary classification of **real** and **AI-generated** images.

| Split | Real Images | Fake Images | Total |
|---|---:|---:|---:|
| Train | 50,000 | 50,000 | 100,000 |
| Test | 10,000 | 10,000 | 20,000 |
| Total | 60,000 | 60,000 | 120,000 |

### Expected dataset structure

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

> Note: The dataset is **not included** in this repository. Please download CIFAKE separately.

---

## Main Experimental Results

### Clean-Test Benchmark on CIFAKE

| Rank | Model | Best Epoch | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) | AUC |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | **AG-EfficientViT V3** | 10 | **98.865** | **98.938** | 98.790 | **98.864** | **0.999116** |
| 2 | ViT-Tiny | 20 | 98.750 | 98.809 | 98.690 | 98.749 | 0.998620 |
| 3 | EfficientViT-Hybrid | 15 | 98.710 | 98.778 | 98.640 | 98.709 | 0.998759 |
| 4 | AG-EfficientViT V1 | 13 | 98.670 | 98.807 | 98.530 | 98.668 | 0.998713 |
| 5 | AG-EfficientViT V2 | 18 | 98.625 | 98.225 | **99.040** | 98.631 | 0.998733 |
| 6 | EfficientNetB0 | 19 | 98.065 | 98.214 | 97.910 | 98.062 | 0.997674 |

### Key observation

The results show that **AG-EfficientViT V3** achieves the strongest overall performance on the clean CIFAKE test set. Compared with the strongest single-branch baseline, **ViT-Tiny**, the proposed model improves:

- **Accuracy:** 98.750% → **98.865%**
- **F1-score:** 98.749% → **98.864%**
- **AUC:** 0.998620 → **0.999116**

---

## Figure 2. Main Results Comparison

> Replace the placeholder below with your final performance comparison figure (bar chart or summary plot).

<p align="center">
  <img src="docs/figures/main_results_comparison.png" alt="Main Results Comparison" width="900"/>
</p>

**Figure 2.** Comparison of all evaluated models on the CIFAKE clean test set.

---

## Ablation Study

To analyze the contribution of each design choice, we evaluated several baselines and intermediate variants.

### Ablation Table

| Variant | CNN Branch | ViT Branch | Artifact Branch | Fusion Strategy | Accuracy (%) | F1-score (%) | AUC |
|---|:---:|:---:|:---:|---|---:|---:|---:|
| EfficientNetB0 | ✓ | - | - | - | 98.065 | 98.062 | 0.997674 |
| ViT-Tiny | - | ✓ | - | - | 98.750 | 98.749 | 0.998620 |
| EfficientViT-Hybrid | ✓ | ✓ | - | Feature Concatenation | 98.710 | 98.709 | 0.998759 |
| AG-EfficientViT V1 | ✓ | ✓ | ✓ | Weighted Fusion | 98.670 | 98.668 | 0.998713 |
| AG-EfficientViT V2 | ✓ | ✓ | ✓ | Gated Concatenation | 98.625 | 98.631 | 0.998733 |
| **AG-EfficientViT V3** | ✓ | ✓ | ✓ | **Logit-Level Fusion** | **98.865** | **98.864** | **0.999116** |

### Ablation Discussion

The ablation results provide three important findings:

1. **ViT-Tiny is a very strong single-branch baseline**, outperforming EfficientNetB0 by a clear margin.
2. **Naïve hybridization is not sufficient**. The simple EfficientViT-Hybrid model (CNN + ViT concatenation) does not surpass the final proposed model.
3. **Fusion design matters significantly**. Although AG-EfficientViT V1 and V2 introduce artifact-aware hybridization, the strongest configuration is achieved only in **AG-EfficientViT V3**, which combines:
   - fine-tuned branch initialization, and
   - artifact-guided logit-level fusion.

### Figure 3. Ablation Visualization

> Replace the placeholder below with your final ablation figure or a rendered table figure.

<p align="center">
  <img src="docs/figures/ablation_results.png" alt="Ablation Results" width="900"/>
</p>

**Figure 3.** Ablation comparison across baselines and AG-EfficientViT variants.

---

## Robustness Analysis

We evaluated model robustness under several common image degradation conditions:

- **JPEG compression**: Q70, Q50, Q30
- **Blur**: radius 1 and radius 2
- **Resize degradation**
- **Additive noise**

### Summary of robustness observations

- **AG-EfficientViT V3** performs best under:
  - **clean**
  - **jpeg_q70**
  - **jpeg_q50**
  - **jpeg_q30**
- **EfficientViT-Hybrid** shows stronger behavior under:
  - stronger **blur**
- **ViT-Tiny** remains more stable under:
  - **resize degradation**
  - **noise corruption**

### Interpretation

These results suggest that AG-EfficientViT V3 provides the best performance in **clean** and **JPEG-compressed** conditions, but **severe blur, resize degradation, and additive noise remain challenging**. Therefore, the robustness claim should be interpreted carefully and **degradation-specifically**, rather than as universal robustness to all corruptions.

### Figure 4. Robustness Plot

> Replace the placeholder below with your robustness plot generated from `robustness_cifake_key_models.csv`.

<p align="center">
  <img src="docs/figures/robustness_plot.png" alt="Robustness Plot" width="950"/>
</p>

**Figure 4.** Performance comparison of key models under different image degradation conditions.

---

## Qualitative Results

This section is intended to show **representative predictions**, including:
- correctly classified **real** images,
- correctly classified **AI-generated** images,
- **failure cases**, and
- examples where AG-EfficientViT V3 outperforms the baselines.

> Replace the placeholder below with your qualitative result grid.

<p align="center">
  <img src="docs/figures/qualitative_results.png" alt="Qualitative Results" width="950"/>
</p>

**Figure 5.** Representative qualitative examples on CIFAKE, including correct predictions and failure cases.

### Template for qualitative case table

| Case ID | Image Type | Ground Truth | ViT-Tiny | EfficientViT-Hybrid | AG-EfficientViT V3 | Notes |
|---|---|---|---|---|---|---|
| Q1 | Real | Real | Correct | Correct | Correct | Easy sample |
| Q2 | Fake | Fake | Correct | Incorrect | Correct | Artifact branch helps |
| Q3 | Fake | Fake | Incorrect | Incorrect | Correct | Global + artifact cues help |
| Q4 | Real | Real | Correct | Correct | Incorrect | Failure case |
| Q5 | Fake | Fake | Correct | Correct | Incorrect | Hard sample / ambiguous texture |

### Suggested qualitative discussion

The qualitative results should highlight both **success cases** and **failure cases**. In particular, the most informative examples are those where:
- AG-EfficientViT V3 correctly detects synthetic content missed by other baselines, and
- AG-EfficientViT V3 fails under challenging blur/noise conditions.

These examples help explain not only **where the model works well**, but also **where further improvement is needed**.

---

## Visual Interpretability

To better understand the behavior of the proposed model, interpretability analysis such as **Grad-CAM**, **attention visualization**, or **artifact activation maps** can be included.

> Replace the placeholder below with your Grad-CAM or attention visualization.

<p align="center">
  <img src="docs/figures/gradcam_examples.png" alt="Grad-CAM Visualization" width="950"/>
</p>

**Figure 6.** Visual interpretability examples showing regions emphasized by AG-EfficientViT V3.

### Suggested analysis points

- Does the model focus on **unnatural textures**, **face regions**, **edge inconsistencies**, or **background artifacts**?
- Do the highlighted areas differ from ViT-Tiny or EfficientViT-Hybrid?
- Are failure cases associated with diffuse or misleading activation patterns?

---

## Repository Structure

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

---

## Installation

Create and activate a Python environment:

```bash
conda create -n ag_efficientvit python=3.11 -y
conda activate ag_efficientvit
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Training

### EfficientNetB0 baseline
```bash
python train.py --epochs 20 --batch-size 128
```

### ViT-Tiny baseline
```bash
python train_vit.py --epochs 20 --batch-size 128
```

### EfficientViT hybrid baseline
```bash
python train_efficientvit.py --epochs 20 --batch-size 64
```

### AG-EfficientViT V3
```bash
python train_ag_efficientvit_v3.py --epochs 10 --batch-size 128 --lr 0.0001 --weight-decay 0.0001
```

---

## Evaluation

### Evaluate AG-EfficientViT V3
```bash
python evaluate_ag_efficientvit_v3.py --checkpoint checkpoints/ag_efficientvit_v3_cifake_best.pth --batch-size 128
```

### Summarize all model results
```bash
python summarize_results.py
```

### Run robustness evaluation for key models
```bash
python robustness_test.py --models key --batch-size 128
```

---

## Final Proposed Model

The final proposed model in this repository is:

```text
AG-EfficientViT V3
```

It combines:

```text
Fine-tuned EfficientNetB0 checkpoint
+ Fine-tuned ViT-Tiny checkpoint
+ Artifact branch
+ Logit-level fusion
```

The previous **V4** robustness-aware attempt was removed due to **numerical instability** and **collapsed training behavior**, and is therefore **not included** as a final candidate model.

---

## Limitations

This project currently focuses on the **CIFAKE** dataset. Although AG-EfficientViT V3 achieves strong performance in clean and JPEG-compressed settings, the following limitations remain:

- external dataset validation is still needed,
- severe blur and additive noise remain challenging,
- robustness claims should be interpreted carefully,
- qualitative and interpretability analyses should be expanded for stronger scientific evidence.

---

## Future Work

Potential next steps include:

- cross-dataset validation on external AI-generated image datasets,
- stronger qualitative and failure-case analysis,
- Grad-CAM and attention-based explainability,
- calibration and threshold analysis,
- deployment-oriented efficiency benchmarking,
- more stable robustness-aware training strategies.

---

## Citation

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

## Author

**Mochamad Rizal Fauzan**  
AI Engineer | Computer Vision Researcher | Embedded Systems and LLM Researcher

---

## Contact

For academic discussion, collaboration, or repository-related questions, please open an issue or contact the repository author through GitHub.
