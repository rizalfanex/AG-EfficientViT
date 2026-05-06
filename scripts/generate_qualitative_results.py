from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models.vit_baseline import ViTBaseline
from models.efficientvit_baseline import EfficientViTBaseline
from models.ag_efficientvit_v3 import AGEfficientViTV3


FIG_DIR = Path("docs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


class PathImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, target, path


def get_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    return model


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    results = {}

    for images, labels, paths in tqdm(loader, desc=f"Evaluating {model.__class__.__name__}", leave=False):
        images = images.to(device, non_blocking=True)

        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)

        for i, path in enumerate(paths):
            results[path] = {
                "pred": int(preds[i].cpu().item()),
                "conf": float(confs[i].cpu().item()),
            }

    return results


def choose_examples(entries):
    used = set()
    selected = []

    def pick(candidates, name):
        for item in candidates:
            if item["path"] not in used:
                used.add(item["path"])
                item = dict(item)
                item["case_name"] = name
                selected.append(item)
                return True
        return False

    # Easy real
    easy_real = [e for e in entries if e["gt_name"] == "real" and e["vit_correct"] and e["hybrid_correct"] and e["v3_correct"]]
    easy_real = sorted(easy_real, key=lambda x: x["v3_conf"], reverse=True)
    pick(easy_real, "Easy Real")

    # Easy fake
    easy_fake = [e for e in entries if e["gt_name"] == "fake" and e["vit_correct"] and e["hybrid_correct"] and e["v3_correct"]]
    easy_fake = sorted(easy_fake, key=lambda x: x["v3_conf"], reverse=True)
    pick(easy_fake, "Easy Fake")

    # V3 wins
    v3_wins = [
        e for e in entries
        if e["v3_correct"] and ((not e["vit_correct"]) or (not e["hybrid_correct"]))
    ]
    v3_wins = sorted(v3_wins, key=lambda x: x["v3_conf"], reverse=True)

    pick(v3_wins, "V3 Win 1")
    pick(v3_wins, "V3 Win 2")

    # Failure cases
    v3_fail = [e for e in entries if not e["v3_correct"]]
    v3_fail = sorted(v3_fail, key=lambda x: x["v3_conf"], reverse=True)

    pick(v3_fail, "V3 Failure 1")
    pick(v3_fail, "V3 Failure 2")

    # Fallback if not enough
    if len(selected) < 6:
        fallback = sorted(entries, key=lambda x: x["v3_conf"], reverse=True)
        for idx, e in enumerate(fallback):
            if len(selected) >= 6:
                break
            if e["path"] not in used:
                e = dict(e)
                e["case_name"] = f"Sample {len(selected)+1}"
                selected.append(e)
                used.add(e["path"])

    return selected[:6]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path("data/CIFAKE/test")

    dataset = PathImageFolder(data_root, transform=get_transform(224))
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    vit = ViTBaseline(num_classes=2, pretrained=False).to(device)
    hybrid = EfficientViTBaseline(num_classes=2, pretrained=False).to(device)
    v3 = AGEfficientViTV3(num_classes=2, pretrained=False).to(device)

    load_checkpoint(vit, "checkpoints/vit_tiny_cifake_best.pth", device)
    load_checkpoint(hybrid, "checkpoints/efficientvit_baseline_cifake_best.pth", device)
    load_checkpoint(v3, "checkpoints/ag_efficientvit_v3_cifake_best.pth", device)

    vit_results = evaluate_model(vit, loader, device)
    hybrid_results = evaluate_model(hybrid, loader, device)
    v3_results = evaluate_model(v3, loader, device)

    entries = []
    for path, gt in dataset.samples:
        gt_name = idx_to_class[gt]

        vit_pred = vit_results[path]["pred"]
        hybrid_pred = hybrid_results[path]["pred"]
        v3_pred = v3_results[path]["pred"]

        entry = {
            "path": path,
            "gt": gt,
            "gt_name": gt_name,
            "vit_pred": vit_pred,
            "hybrid_pred": hybrid_pred,
            "v3_pred": v3_pred,
            "vit_name": idx_to_class[vit_pred],
            "hybrid_name": idx_to_class[hybrid_pred],
            "v3_name": idx_to_class[v3_pred],
            "vit_conf": vit_results[path]["conf"],
            "hybrid_conf": hybrid_results[path]["conf"],
            "v3_conf": v3_results[path]["conf"],
            "vit_correct": vit_pred == gt,
            "hybrid_correct": hybrid_pred == gt,
            "v3_correct": v3_pred == gt,
        }
        entries.append(entry)

    selected = choose_examples(entries)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax, item in zip(axes, selected):
        img = Image.open(item["path"]).convert("RGB").resize((224, 224))
        ax.imshow(img)
        ax.axis("off")

        title = (
            f"{item['case_name']}\n"
            f"GT: {item['gt_name']}\n"
            f"ViT: {item['vit_name']} ({item['vit_conf']:.2f}) | "
            f"Hybrid: {item['hybrid_name']} ({item['hybrid_conf']:.2f})\n"
            f"V3: {item['v3_name']} ({item['v3_conf']:.2f})"
        )
        ax.set_title(title, fontsize=9)

    for i in range(len(selected), len(axes)):
        axes[i].axis("off")

    fig.suptitle("Qualitative Results on CIFAKE", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = FIG_DIR / "qualitative_results.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
