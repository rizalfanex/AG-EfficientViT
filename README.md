# Artifact Guided EfficientViT: A Robust Hybrid CNN and Transformer Model for AI Generated Image Detection

<p align="center">
  <b>Artifact Guided EfficientViT (AG-EfficientViT)</b><br>
  A hybrid CNN and Transformer framework for real vs AI-generated image detection.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Task-AI%20Generated%20Image%20Detection-blue">
  <img src="https://img.shields.io/badge/Dataset-CIFAKE-green">
  <img src="https://img.shields.io/badge/Framework-PyTorch-red">
  <img src="https://img.shields.io/badge/Model-AG--EfficientViT%20V3-purple">
</p>

---

## Abstract

The rapid progress of generative artificial intelligence has increased the difficulty of distinguishing real images from AI-generated content. Although convolutional neural networks and Vision Transformers have shown strong performance in visual classification tasks, single-branch models may not fully capture both local artifact patterns and global semantic inconsistencies. This repository presents **Artifact Guided EfficientViT**, a hybrid CNN and Transformer framework for AI-generated image detection. The final proposed model, **AG-EfficientViT V3**, integrates a fine-tuned EfficientNetB0 branch, a fine-tuned ViT-Tiny branch, and an artifact-oriented branch through logit-level fusion. Experiments on the CIFAKE dataset show that AG-EfficientViT V3 achieves the best clean test performance among all evaluated baselines, reaching **98.865% accuracy**, **98.864% F1-score**, and **0.999116 AUC**. The results indicate that fine-tuned branch initialization and artifact-guided fusion provide a more effective detection framework than simple CNN-Transformer concatenation.

---

## Highlights

- Proposes **AG-EfficientViT V3**, an artifact-guided hybrid CNN and Transformer model.
- Combines **EfficientNetB0**, **ViT-Tiny**, and an **artifact branch** for binary real/fake image classification.
- Uses **fine-tuned branch initialization** from the best CNN and Transformer checkpoints.
- Achieves the best clean CIFAKE performance among all tested models.
- Provides baseline comparison, ablation variants, clean evaluation, and robustness analysis.

---

## Method Overview

The final model, **AG-EfficientViT V3**, consists of four main components:

1. **EfficientNetB0 Branch**  
   Captures local texture, edge, and spatial artifact patterns.

2. **ViT-Tiny Branch**  
   Captures global semantic representation and long-range visual dependencies.

3. **Artifact Branch**  
   Extracts high-pass residual features related to synthetic image traces.

4. **Logit-Level Fusion**  
   Combines fine-tuned CNN logits, fine-tuned Transformer logits, and artifact logits for final prediction.

```text
Input Image
    │
    ├── Fine-tuned EfficientNetB0 Branch
    │       └── CNN logits
    │
    ├── Fine-tuned ViT-Tiny Branch
    │       └── Transformer logits
    │
    ├── Artifact Branch
    │       └── Artifact logits
    │
    └── Logit-Level Fusion
            └── Real / AI Generated
```

---

## Dataset

This project uses the **CIFAKE** dataset for real vs AI-generated image detection.

| Split | Real Images | Fake Images | Total |
|---|---:|---:|---:|
| Train | 50,000 | 50,000 | 100,000 |
| Test | 10,000 | 10,000 | 20,000 |
| Total | 60,000 | 60,000 | 120,000 |

The dataset is not included in this repository. Users should download CIFAKE separately and organize it as follows:

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

---

## Experimental Results on CIFAKE

| Rank | Model | Best Epoch | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) | AUC |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | **AG-EfficientViT V3** | 10 | **98.865** | **98.938** | 98.790 | **98.864** | **0.999116** |
| 2 | ViT-Tiny | 20 | 98.750 | 98.809 | 98.690 | 98.749 | 0.998620 |
| 3 | EfficientViT-Hybrid | 15 | 98.710 | 98.778 | 98.640 | 98.709 | 0.998759 |
| 4 | AG-EfficientViT V1 | 13 | 98.670 | 98.807 | 98.530 | 98.668 | 0.998713 |
| 5 | AG-EfficientViT V2 | 18 | 98.625 | 98.225 | **99.040** | 98.631 | 0.998733 |
| 6 | EfficientNetB0 | 19 | 98.065 | 98.214 | 97.910 | 98.062 | 0.997674 |

---

## Key Findings

The results show that **AG-EfficientViT V3** achieves the strongest overall clean CIFAKE performance. Compared with the strongest single baseline, ViT-Tiny, AG-EfficientViT V3 improves accuracy from **98.750%** to **98.865%** and AUC from **0.998620** to **0.999116**.

The experiments also show that simple CNN and Transformer concatenation does not automatically improve performance. The EfficientViT-Hybrid baseline reaches **98.710% accuracy**, which is lower than the ViT-Tiny baseline. This supports the need for a more careful fusion design using fine-tuned branch initialization and artifact-guided correction.

---

## Robustness Analysis

Robustness evaluation was conducted under several image degradation conditions, including JPEG compression, Gaussian blur, resize degradation, and additive noise.

AG-EfficientViT V3 performs strongly under **clean** and **JPEG-compressed** conditions. However, severe blur, resize degradation, and additive noise remain challenging. Therefore, the robustness claim should be interpreted as degradation-specific rather than universal robustness across all corruptions.

| Condition | Best Observation |
|---|---|
| Clean | AG-EfficientViT V3 achieves the best performance |
| JPEG Compression | AG-EfficientViT V3 remains strongest across JPEG Q70, Q50, and Q30 |
| Blur | EfficientViT-Hybrid is more stable under stronger blur |
| Resize Degradation | ViT-Tiny remains competitive |
| Additive Noise | ViT-Tiny is more robust than hybrid variants |

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

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

Train EfficientNetB0 baseline:

```bash
python train.py --epochs 20 --batch-size 128
```

Train ViT-Tiny baseline:

```bash
python train_vit.py --epochs 20 --batch-size 128
```

Train EfficientViT hybrid baseline:

```bash
python train_efficientvit.py --epochs 20 --batch-size 64
```

Train AG-EfficientViT V3:

```bash
python train_ag_efficientvit_v3.py --epochs 10 --batch-size 128 --lr 0.0001 --weight-decay 0.0001
```

---

## Evaluation

Evaluate AG-EfficientViT V3:

```bash
python evaluate_ag_efficientvit_v3.py --checkpoint checkpoints/ag_efficientvit_v3_cifake_best.pth --batch-size 128
```

Summarize all model results:

```bash
python summarize_results.py
```

Run robustness evaluation for key models:

```bash
python robustness_test.py --models key --batch-size 128
```

---

## Final Proposed Model

The final proposed model is:

```text
AG-EfficientViT V3
```

AG-EfficientViT V3 uses:

```text
Fine-tuned EfficientNetB0 checkpoint
+ Fine-tuned ViT-Tiny checkpoint
+ Artifact branch
+ Logit-level fusion
```

The previous V4 robustness-aware attempt was removed due to numerical instability and collapsed training behavior.

---

## Limitations

This project currently focuses on the CIFAKE dataset. Although the proposed model achieves strong clean and JPEG-compressed performance, additional validation on external datasets is required to support stronger claims of generalization. Severe blur, resize degradation, and additive noise remain challenging and should be further investigated in future work.

---

## Future Work

- Cross-dataset validation on external AI-generated image datasets.
- Grad-CAM and attention-based explainability.
- Calibration analysis and threshold optimization.
- Robustness-aware training with stable corruption augmentation.
- Deployment-oriented model compression and inference benchmarking.

---

## Citation

If you find this repository useful, please cite it as:

```bibtex
@misc{ag_efficientvit_2026,
  title  = {Artifact Guided EfficientViT: A Robust Hybrid CNN and Transformer Model for AI Generated Image Detection},
  author = {Mochamad Rizal Fauzan},
  year   = {2026},
  note   = {Research implementation for AI-generated image detection}
}
```

---

## Author

**Mochamad Rizal Fauzan**  
AI Engineer | Computer Vision Researcher | Embedded Systems and LLM Researcher
