import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

from datasets.cifake_dataset import build_cifake_loaders
from models.efficientvit_baseline import EfficientViTBaseline


@torch.no_grad()
def run_evaluation(model, loader, device, use_amp=True):
    model.eval()

    all_true = []
    all_pred = []
    all_prob = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        all_true.extend(labels.detach().cpu().numpy().tolist())
        all_pred.extend(preds.detach().cpu().numpy().tolist())
        all_prob.extend(probs.detach().cpu().numpy().tolist())

    return np.array(all_true), np.array(all_pred), np.array(all_prob)


def save_metrics_csv(path, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def plot_confusion_matrix(cm, classes, save_path):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)

    ax.set_title("Confusion Matrix - EfficientViT-Hybrid Baseline")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curve(y_true, y_prob, save_path):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - EfficientViT-Hybrid Baseline")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/efficientvit_baseline_cifake_best.pth")
    parser.add_argument("--data-root", type=str, default="data/CIFAKE")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available() and not args.no_amp

    print("=" * 80)
    print("Evaluating EfficientViT Hybrid Baseline")
    print("=" * 80)
    print(f"Device     : {device}")
    print(f"GPU        : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Checkpoint : {args.checkpoint}")
    print("=" * 80)

    _, test_loader, classes = build_cifake_loaders(
        root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = EfficientViTBaseline(num_classes=2, pretrained=False).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    y_true, y_pred, y_prob = run_evaluation(
        model=model,
        loader=test_loader,
        device=device,
        use_amp=use_amp,
    )

    acc = accuracy_score(y_true, y_pred)

    precision_binary, recall_binary, f1_binary, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    auc_score = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "model": "EfficientViT-Hybrid",
        "dataset": "CIFAKE",
        "checkpoint_epoch": checkpoint.get("epoch", "unknown"),
        "accuracy": acc,
        "precision_binary": precision_binary,
        "recall_binary": recall_binary,
        "f1_binary": f1_binary,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "auc": auc_score,
    }

    save_metrics_csv(
        Path("results/tables/efficientvit_baseline_cifake_metrics.csv"),
        metrics,
    )

    plot_confusion_matrix(
        cm,
        classes,
        Path("results/figures/efficientvit_baseline_confusion_matrix.png"),
    )

    plot_roc_curve(
        y_true,
        y_prob,
        Path("results/figures/efficientvit_baseline_roc_curve.png"),
    )

    print("\nFinal Metrics")
    print("-" * 80)
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\nClassification Report")
    print("-" * 80)
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))

    print("\nSaved:")
    print("results/tables/efficientvit_baseline_cifake_metrics.csv")
    print("results/figures/efficientvit_baseline_confusion_matrix.png")
    print("results/figures/efficientvit_baseline_roc_curve.png")


if __name__ == "__main__":
    main()

