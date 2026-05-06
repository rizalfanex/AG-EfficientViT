from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np


FIG_DIR = Path("docs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def save_architecture_figure():
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#F8FAFC", ec="#1E293B", fontsize=11, bold=False):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.03,rounding_size=0.12",
            linewidth=1.8,
            edgecolor=ec,
            facecolor=fc
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=fontsize,
            fontweight="bold" if bold else "normal",
            color="#0F172A",
            wrap=True
        )

    def arrow(x1, y1, x2, y2):
        ax.annotate(
            "",
            xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", lw=1.8, color="#334155")
        )

    box(0.5, 3.25, 2.0, 1.0, "Input Image\n224 × 224 × 3", fc="#E0F2FE", ec="#0284C7", bold=True)

    box(3.4, 5.8, 3.0, 1.0, "Fine-tuned\nEfficientNetB0 Branch", fc="#DCFCE7", ec="#16A34A", bold=True)
    box(3.4, 3.25, 3.0, 1.0, "Fine-tuned\nViT-Tiny Branch", fc="#EDE9FE", ec="#7C3AED", bold=True)
    box(3.4, 0.7, 3.0, 1.0, "Artifact Branch\nHigh-pass Residual Cues", fc="#FEF3C7", ec="#D97706", bold=True)

    box(7.2, 5.8, 2.2, 1.0, "CNN Logits", fc="#F0FDF4", ec="#16A34A")
    box(7.2, 3.25, 2.2, 1.0, "Transformer Logits", fc="#F5F3FF", ec="#7C3AED")
    box(7.2, 0.7, 2.2, 1.0, "Artifact Logits", fc="#FFFBEB", ec="#D97706")

    box(10.2, 3.25, 2.2, 1.1, "Logit-Level\nFusion", fc="#FFE4E6", ec="#E11D48", bold=True)
    box(13.0, 3.25, 1.6, 1.1, "Real / AI\nGenerated", fc="#F1F5F9", ec="#0F172A", bold=True)

    arrow(2.5, 3.75, 3.4, 6.3)
    arrow(2.5, 3.75, 3.4, 3.75)
    arrow(2.5, 3.75, 3.4, 1.2)

    arrow(6.4, 6.3, 7.2, 6.3)
    arrow(6.4, 3.75, 7.2, 3.75)
    arrow(6.4, 1.2, 7.2, 1.2)

    arrow(9.4, 6.3, 10.2, 3.9)
    arrow(9.4, 3.75, 10.2, 3.75)
    arrow(9.4, 1.2, 10.2, 3.6)

    arrow(12.4, 3.8, 13.0, 3.8)

    ax.text(
        7.5, 7.45,
        "AG-EfficientViT V3: Fine-Tuned Branch Initialization with Artifact-Guided Logit Fusion",
        ha="center", va="center",
        fontsize=16,
        fontweight="bold",
        color="#0F172A"
    )

    ax.text(
        7.5, 0.15,
        "Final CIFAKE Result: Accuracy 98.865% | F1-score 98.864% | AUC 0.999116",
        ha="center", va="center",
        fontsize=11,
        color="#334155"
    )

    out = FIG_DIR / "ag_efficientvit_v3_architecture.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def save_main_results_plot():
    path = Path("results/tables/final_model_comparison_cifake.csv")
    df = pd.read_csv(path)

    df = df.sort_values("Accuracy (%)", ascending=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(df["Model"], df["Accuracy (%)"])

    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Clean CIFAKE Performance Comparison")
    ax.set_xlim(97.8, 99.0)

    for bar, value in zip(bars, df["Accuracy (%)"]):
        ax.text(value + 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.3f}%", va="center", fontsize=10)

    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()

    out = FIG_DIR / "main_results_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def save_ablation_plot():
    rows = [
        ["EfficientNetB0", 98.065, 98.062, 0.997674],
        ["ViT-Tiny", 98.750, 98.749, 0.998620],
        ["EfficientViT-Hybrid", 98.710, 98.709, 0.998759],
        ["AG-EfficientViT V1", 98.670, 98.668, 0.998713],
        ["AG-EfficientViT V2", 98.625, 98.631, 0.998733],
        ["AG-EfficientViT V3", 98.865, 98.864, 0.999116],
    ]

    df = pd.DataFrame(rows, columns=["Variant", "Accuracy", "F1-score", "AUC"])

    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, df["Accuracy"], width, label="Accuracy")
    ax.bar(x + width / 2, df["F1-score"], width, label="F1-score")

    ax.set_title("Ablation Study on CIFAKE")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(97.8, 99.0)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Variant"], rotation=25, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    for i, row in df.iterrows():
        ax.text(i - width / 2, row["Accuracy"] + 0.01, f"{row['Accuracy']:.3f}", ha="center", fontsize=8)
        ax.text(i + width / 2, row["F1-score"] + 0.01, f"{row['F1-score']:.3f}", ha="center", fontsize=8)

    plt.tight_layout()

    out = FIG_DIR / "ablation_results.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def save_robustness_plot():
    path = Path("results/tables/robustness_cifake_key_models.csv")
    df = pd.read_csv(path)

    df["accuracy_percent"] = df["accuracy"] * 100

    conditions = [
        "clean",
        "jpeg_q70",
        "jpeg_q50",
        "jpeg_q30",
        "blur_r1",
        "blur_r2",
        "resize_112",
        "noise_003",
        "noise_005",
    ]

    fig, ax = plt.subplots(figsize=(13, 6))

    for model in df["model"].unique():
        sub = df[df["model"] == model].copy()
        sub["condition"] = pd.Categorical(sub["condition"], categories=conditions, ordered=True)
        sub = sub.sort_values("condition")
        ax.plot(sub["condition"], sub["accuracy_percent"], marker="o", linewidth=2, label=model)

    ax.set_title("Robustness Evaluation under Image Degradation")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Degradation Condition")
    ax.set_ylim(45, 100)
    ax.grid(alpha=0.25)
    ax.legend()
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    out = FIG_DIR / "robustness_plot.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main():
    save_architecture_figure()
    save_main_results_plot()
    save_ablation_plot()
    save_robustness_plot()


if __name__ == "__main__":
    main()
