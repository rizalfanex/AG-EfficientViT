# Artifact Guided EfficientViT

## Project Title

Artifact Guided EfficientViT: A Robust Hybrid CNN and Transformer Model for AI Generated Image Detection

## Final Proposed Model

The final proposed model is AG-EfficientViT V3.

AG-EfficientViT V3 combines:
- Fine-tuned EfficientNetB0 branch
- Fine-tuned ViT-Tiny branch
- Artifact branch
- Logit-level fusion

The model uses fine-tuned branch initialization from the best EfficientNetB0 and ViT-Tiny checkpoints, then learns artifact-guided correction for final binary classification.

## Dataset

Dataset: CIFAKE

Classes:
- fake
- real

Training images:
- 50,000 fake
- 50,000 real

Testing images:
- 10,000 fake
- 10,000 real

Total images:
- 120,000 images

## Final Clean Test Result

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) | AUC |
|---|---:|---:|---:|---:|---:|
| AG-EfficientViT V3 | 98.865 | 98.938 | 98.790 | 98.864 | 0.999116 |
| ViT-Tiny | 98.750 | 98.809 | 98.690 | 98.749 | 0.998620 |
| EfficientViT-Hybrid | 98.710 | 98.778 | 98.640 | 98.709 | 0.998759 |
| AG-EfficientViT V1 | 98.670 | 98.807 | 98.530 | 98.668 | 0.998713 |
| AG-EfficientViT V2 | 98.625 | 98.225 | 99.040 | 98.631 | 0.998733 |
| EfficientNetB0 | 98.065 | 98.214 | 97.910 | 98.062 | 0.997674 |

## Main Finding

AG-EfficientViT V3 achieved the best clean CIFAKE performance among all evaluated models. It improved over the strongest single baseline, ViT-Tiny, and over the simple EfficientViT hybrid baseline.

## Technical Interpretation

The results show that simple CNN and Transformer concatenation is not sufficient to consistently improve performance. AG-EfficientViT V3 improves the hybrid framework by using fine-tuned branch initialization and artifact-guided logit-level fusion.

## Robustness Finding

AG-EfficientViT V3 performed strongly under clean and JPEG-compressed conditions. However, severe blur, resize degradation, and additive noise remain challenging. Therefore, the robustness claim should be written carefully and supported by detailed degradation-specific analysis.

## Final Decision

Final proposed model:
AG-EfficientViT V3

Dropped model:
AG-EfficientViT V4, due to numerical instability and collapsed training.

## Next Paper Tasks

1. Prepare architecture figure.
2. Prepare final result table.
3. Prepare robustness table.
4. Prepare ablation discussion.
5. Add Grad-CAM or attention visualization.
6. Add external dataset validation for stronger journal submission.
7. Draft the manuscript.
