from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

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
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    return model


@torch.no_grad()
def collect_candidates(model, loader, device, idx_to_class):
    model.eval()
    rows = []

    for images, labels, paths in tqdm(loader, desc="Selecting Grad-CAM samples", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)

        for i, path in enumerate(paths):
            gt = int(labels[i].item())
            pred = int(preds[i].cpu().item())
            conf = float(confs[i].cpu().item())
            rows.append({
                "path": path,
                "gt": gt,
                "pred": pred,
                "conf": conf,
                "correct": gt == pred,
                "gt_name": idx_to_class[gt],
                "pred_name": idx_to_class[pred],
            })

    real_correct = sorted(
        [r for r in rows if r["gt_name"] == "real" and r["correct"]],
        key=lambda x: x["conf"],
        reverse=True,
    )

    fake_correct = sorted(
        [r for r in rows if r["gt_name"] == "fake" and r["correct"]],
        key=lambda x: x["conf"],
        reverse=True,
    )

    selected = []
    selected.extend(real_correct[:2])
    selected.extend(fake_correct[:2])

    if len(selected) < 4:
        fallback = sorted(
            [r for r in rows if r["correct"]],
            key=lambda x: x["conf"],
            reverse=True,
        )
        used = {x["path"] for x in selected}
        for item in fallback:
            if len(selected) >= 4:
                break
            if item["path"] not in used:
                selected.append(item)
                used.add(item["path"])

    return selected[:4]


def compute_gradcam(model, input_tensor, target_class, target_layer):
    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad(set_to_none=True)
    output = model(input_tensor)
    score = output[:, target_class].sum()
    score.backward()

    acts = activations[0]
    grads = gradients[0]

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)

    cam = cam[0, 0].detach().cpu().numpy()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    h1.remove()
    h2.remove()

    return cam


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path("data/CIFAKE/test")

    dataset = PathImageFolder(data_root, transform=get_transform(224))
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    model = AGEfficientViTV3(
        num_classes=2,
        pretrained=False,
        freeze_backbones=False,
    ).to(device)

    load_checkpoint(
        model,
        "checkpoints/ag_efficientvit_v3_cifake_best.pth",
        device,
    )

    model.eval()

    selected = collect_candidates(model, loader, device, idx_to_class)

    target_layer = model.cnn_model.conv_head

    fig, axes = plt.subplots(len(selected), 2, figsize=(10, 4 * len(selected)))

    if len(selected) == 1:
        axes = np.array([axes])

    for row_idx, item in enumerate(selected):
        raw_img = Image.open(item["path"]).convert("RGB").resize((224, 224))
        input_tensor = get_transform(224)(raw_img).unsqueeze(0).to(device)

        cam = compute_gradcam(
            model=model,
            input_tensor=input_tensor,
            target_class=item["pred"],
            target_layer=target_layer,
        )

        axes[row_idx, 0].imshow(raw_img)
        axes[row_idx, 0].axis("off")
        axes[row_idx, 0].set_title(
            f"Original\nGT: {item['gt_name']} | Pred: {item['pred_name']} ({item['conf']:.2f})",
            fontsize=10,
        )

        axes[row_idx, 1].imshow(raw_img)
        axes[row_idx, 1].imshow(cam, cmap="jet", alpha=0.4)
        axes[row_idx, 1].axis("off")
        axes[row_idx, 1].set_title("Grad-CAM Overlay", fontsize=10)

    fig.suptitle(
        "Grad-CAM Examples for AG-EfficientViT V3",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = FIG_DIR / "gradcam_examples.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
