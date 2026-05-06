import pandas as pd
from pathlib import Path

files = [
    "results/tables/efficientnet_b0_cifake_metrics.csv",
    "results/tables/vit_tiny_cifake_metrics.csv",
    "results/tables/efficientvit_baseline_cifake_metrics.csv",
    "results/tables/ag_efficientvit_cifake_metrics.csv",
    "results/tables/ag_efficientvit_v2_cifake_metrics.csv",
    "results/tables/ag_efficientvit_v3_cifake_metrics.csv",
]

rows = []

for file in files:
    path = Path(file)
    if not path.exists():
        print(f"[Warning] Missing: {file}")
        continue

    df = pd.read_csv(path)
    row = df.iloc[0].to_dict()

    rows.append({
        "Model": row.get("model", path.stem),
        "Dataset": row.get("dataset", "CIFAKE"),
        "Best Epoch": row.get("checkpoint_epoch", ""),
        "Accuracy (%)": float(row["accuracy"]) * 100,
        "Precision (%)": float(row["precision_binary"]) * 100,
        "Recall (%)": float(row["recall_binary"]) * 100,
        "F1-score (%)": float(row["f1_binary"]) * 100,
        "AUC": float(row["auc"]),
    })

summary = pd.DataFrame(rows)

summary = summary.sort_values(
    by=["Accuracy (%)", "F1-score (%)", "AUC"],
    ascending=False
)

out_path = Path("results/tables/final_model_comparison_cifake.csv")
summary.to_csv(out_path, index=False)

print(summary.to_string(index=False))
print(f"\nSaved to: {out_path}")
