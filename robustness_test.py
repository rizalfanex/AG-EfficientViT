import argparse
import csv
import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models.vit_baseline import ViTBaseline
from models.efficientvit_baseline import EfficientViTBaseline
from models.ag_efficientvit_v3 import AGEfficientViTV3


class RobustTransform:
    def __init__(self, image_size=224, condition="clean"):
        self.image_size = image_size
        self.condition = condition
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def jpeg_compress(self, img, quality):
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def __call__(self, img):
        img = img.convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

        if self.condition == "jpeg_q70":
            img = self.jpeg_compress(img, 70)
        elif self.condition == "jpeg_q50":
            img = self.jpeg_compress(img, 50)
        elif self.condition == "jpeg_q30":
            img = self.jpeg_compress(img, 30)
        elif self.condition == "blur_r1":
            img = img.filter(ImageFilter.GaussianBlur(radius=1.0))
        elif self.condition == "blur_r2":
            img = img.filter(ImageFilter.GaussianBlur(radius=2.0))
        elif self.condition == "resize_112":
            img = img.resize((112, 112), Image.BILINEAR)
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

        x = transforms.ToTensor()(img)

        if self.condition == "noise_003":
            noise = torch.randn_like(x) * 0.03
            x = torch.clamp(x + noise, 0.0, 1.0)
        elif self.condition == "noise_005":
            noise = torch.randn_like(x) * 0.05
            x = torch.clamp(x + noise, 0.0, 1.0)

        x = self.normalize(x)
        return x


def build_loader(data_root, image_size, batch_size, num_workers, condition):
    dataset = datasets.ImageFolder(
        Path(data_root) / "test",
        transform=RobustTransform(image_size=image_size, condition=condition),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader, dataset.classes


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    return checkpoint


@torch.no_grad()
def evaluate_model(model, loader, device, use_amp=True, max_batches=None):
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    for batch_idx, (images, labels) in enumerate(tqdm(loader, leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())
        y_prob.extend(probs.detach().cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.0

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }


def build_models(device, model_group):
    registry = {
        "vit": {
            "name": "ViT-Tiny",
            "model": ViTBaseline(num_classes=2, pretrained=False),
            "checkpoint": "checkpoints/vit_tiny_cifake_best.pth",
        },
        "hybrid": {
            "name": "EfficientViT-Hybrid",
            "model": EfficientViTBaseline(num_classes=2, pretrained=False),
            "checkpoint": "checkpoints/efficientvit_baseline_cifake_best.pth",
        },
        "v3": {
            "name": "AG-EfficientViT-V3",
            "model": AGEfficientViTV3(num_classes=2, pretrained=False),
            "checkpoint": "checkpoints/ag_efficientvit_v3_cifake_best.pth",
        },
    }

    if model_group == "proposed":
        selected = ["v3"]
    else:
        selected = ["vit", "hybrid", "v3"]

    models = []

    for key in selected:
        item = registry[key]
        model = item["model"].to(device)
        load_checkpoint(model, item["checkpoint"], device)
        models.append((item["name"], model))

    return models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/CIFAKE")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--models", type=str, default="key", choices=["key", "proposed"])
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()

    print("=" * 80)
    print("Robustness Test | CIFAKE")
    print("=" * 80)
    print(f"Device     : {device}")
    print(f"GPU        : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Models     : {args.models}")
    print(f"Batch size : {args.batch_size}")
    print("=" * 80)

    models = build_models(device, args.models)

    rows = []

    for condition in conditions:
        print(f"\nCondition: {condition}")
        loader, classes = build_loader(
            data_root=args.data_root,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            condition=condition,
        )

        for model_name, model in models:
            print(f"Evaluating: {model_name}")
            metrics = evaluate_model(
                model=model,
                loader=loader,
                device=device,
                use_amp=use_amp,
                max_batches=args.max_batches,
            )

            row = {
                "model": model_name,
                "condition": condition,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "auc": metrics["auc"],
            }

            rows.append(row)

            print(
                f"{model_name} | "
                f"Acc: {metrics['accuracy']:.4f} | "
                f"F1: {metrics['f1']:.4f} | "
                f"AUC: {metrics['auc']:.4f}"
            )

    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "robustness_cifake_key_models.csv"

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "condition", "accuracy", "precision", "recall", "f1", "auc"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved robustness results:")
    print(out_path)


if __name__ == "__main__":
    main()
