import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm

from datasets.cifake_dataset import build_cifake_loaders
from models.vit_baseline import ViTBaseline


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true, y_pred, y_prob):
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


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, max_batches=None):
    model.train()

    total_loss = 0.0
    all_true = []
    all_pred = []
    all_prob = []

    start_time = time.time()

    progress = tqdm(loader, desc="Training", leave=False)

    for batch_idx, (images, labels) in enumerate(progress):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        probs = torch.softmax(logits.detach(), dim=1)[:, 1]
        preds = torch.argmax(logits.detach(), dim=1)

        total_loss += loss.item() * images.size(0)
        all_true.extend(labels.detach().cpu().numpy().tolist())
        all_pred.extend(preds.cpu().numpy().tolist())
        all_prob.extend(probs.cpu().numpy().tolist())

        progress.set_postfix(loss=f"{loss.item():.4f}")

    epoch_time = time.time() - start_time
    avg_loss = total_loss / max(1, len(all_true))
    metrics = compute_metrics(all_true, all_pred, all_prob)

    return avg_loss, metrics, epoch_time


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp, max_batches=None):
    model.eval()

    total_loss = 0.0
    all_true = []
    all_pred = []
    all_prob = []

    progress = tqdm(loader, desc="Evaluating", leave=False)

    for batch_idx, (images, labels) in enumerate(progress):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        total_loss += loss.item() * images.size(0)
        all_true.extend(labels.detach().cpu().numpy().tolist())
        all_pred.extend(preds.cpu().numpy().tolist())
        all_prob.extend(probs.cpu().numpy().tolist())

    avg_loss = total_loss / max(1, len(all_true))
    metrics = compute_metrics(all_true, all_pred, all_prob)

    return avg_loss, metrics


def save_log_row(log_path, row, write_header=False):
    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "train_precision",
        "train_recall",
        "train_f1",
        "train_auc",
        "test_loss",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_auc",
        "epoch_time_sec",
    ]

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/CIFAKE")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available() and not args.no_amp

    print("=" * 80)
    print("AG-EfficientViT Project | Baseline: ViT-Tiny")
    print("=" * 80)
    print(f"Device       : {device}")
    print(f"CUDA         : {torch.cuda.is_available()}")
    print(f"GPU          : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Data root    : {args.data_root}")
    print(f"Image size   : {args.image_size}")
    print(f"Batch size   : {args.batch_size}")
    print(f"Epochs       : {args.epochs}")
    print(f"AMP          : {use_amp}")
    print("=" * 80)

    train_loader, test_loader, classes = build_cifake_loaders(
        root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"Classes      : {classes}")
    print(f"Train images : {len(train_loader.dataset)}")
    print(f"Test images  : {len(test_loader.dataset)}")

    model = ViTBaseline(num_classes=2, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    checkpoint_dir = Path("checkpoints")
    log_dir = Path("results/logs")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / "vit_tiny_cifake_log.csv"

    if log_path.exists():
        log_path.unlink()

    best_f1 = -1.0
    best_path = checkpoint_dir / "vit_tiny_cifake_best.pth"

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_metrics, epoch_time = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            max_batches=args.max_train_batches,
        )

        test_loss, test_metrics = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            max_batches=args.max_val_batches,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "train_auc": train_metrics["auc"],
            "test_loss": test_loss,
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
            "test_auc": test_metrics["auc"],
            "epoch_time_sec": epoch_time,
        }

        save_log_row(log_path, row, write_header=(epoch == 1))

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Acc: {train_metrics['accuracy']:.4f} | "
            f"F1: {train_metrics['f1']:.4f} | "
            f"AUC: {train_metrics['auc']:.4f}"
        )

        print(
            f"Test  Loss: {test_loss:.4f} | "
            f"Acc: {test_metrics['accuracy']:.4f} | "
            f"F1: {test_metrics['f1']:.4f} | "
            f"AUC: {test_metrics['auc']:.4f}"
        )

        print(f"Epoch time: {epoch_time:.2f} sec")

        if test_metrics["f1"] > best_f1:
            best_f1 = test_metrics["f1"]

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_f1": best_f1,
                    "classes": classes,
                    "args": vars(args),
                },
                best_path,
            )

            print(f"Saved best checkpoint: {best_path} | Best F1: {best_f1:.4f}")

    print("\nTraining finished.")
    print(f"Best checkpoint: {best_path}")
    print(f"Log file       : {log_path}")


if __name__ == "__main__":
    main()

