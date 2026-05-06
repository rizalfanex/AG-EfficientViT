from pathlib import Path

import timm
import torch
import torch.nn as nn

from models.artifact_branch import ArtifactBranch


class AGEfficientViTV3(nn.Module):
    """
    AG-EfficientViT V3: Fine-tuned branch initialization.

    Main idea:
        1. Load fine-tuned EfficientNetB0 classifier.
        2. Load fine-tuned ViT-Tiny classifier.
        3. Add trainable artifact branch.
        4. Learn calibrated logit-level fusion.

    This version preserves the strong decision boundary of the best single models
    while allowing artifact-guided correction.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = False,
        cnn_name: str = "efficientnet_b0",
        vit_name: str = "vit_tiny_patch16_224",
        cnn_checkpoint: str = "checkpoints/efficientnet_b0_cifake_best.pth",
        vit_checkpoint: str = "checkpoints/vit_tiny_cifake_best.pth",
        artifact_dim: int = 128,
        dropout: float = 0.30,
        freeze_backbones: bool = True,
    ):
        super().__init__()

        self.freeze_backbones = freeze_backbones

        self.cnn_model = self._create_classifier_model(
            model_name=cnn_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )

        self.vit_model = self._create_classifier_model(
            model_name=vit_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )

        self.artifact_branch = ArtifactBranch(artifact_dim=artifact_dim)

        self.artifact_classifier = nn.Sequential(
            nn.Linear(artifact_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        # Initialized to strongly preserve ViT, with smaller contribution from CNN
        # and minimal artifact contribution at the beginning.
        self.logit_weights = nn.Parameter(
            torch.tensor([1.0, 2.5, -2.0], dtype=torch.float32)
        )

        self._load_checkpoint_into_model(self.cnn_model, cnn_checkpoint, prefix_to_strip="model.")
        self._load_checkpoint_into_model(self.vit_model, vit_checkpoint, prefix_to_strip="model.")

        if self.freeze_backbones:
            for p in self.cnn_model.parameters():
                p.requires_grad = False
            for p in self.vit_model.parameters():
                p.requires_grad = False

    def _create_classifier_model(self, model_name: str, pretrained: bool, num_classes: int):
        try:
            model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes,
            )
        except Exception as exc:
            print(f"[Warning] Failed to create/load {model_name}: {exc}")
            print("[Info] Falling back to pretrained=False.")
            model = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=num_classes,
            )
        return model

    def _load_checkpoint_into_model(self, model, checkpoint_path: str, prefix_to_strip: str = "model."):
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            print(f"[Warning] Checkpoint not found: {checkpoint_path}")
            return

        print(f"[Info] Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        cleaned_state = {}

        for key, value in state_dict.items():
            if key.startswith(prefix_to_strip):
                new_key = key[len(prefix_to_strip):]
            else:
                new_key = key
            cleaned_state[new_key] = value

        missing, unexpected = model.load_state_dict(cleaned_state, strict=False)

        print(f"[Info] Loaded: {checkpoint_path.name}")
        print(f"[Info] Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")

    def forward(self, x, return_gates: bool = False):
        if self.freeze_backbones:
            self.cnn_model.eval()
            self.vit_model.eval()

            with torch.no_grad():
                cnn_logits = self.cnn_model(x)
                vit_logits = self.vit_model(x)
        else:
            cnn_logits = self.cnn_model(x)
            vit_logits = self.vit_model(x)

        artifact_feat = self.artifact_branch(x)
        artifact_logits = self.artifact_classifier(artifact_feat)

        weights = torch.softmax(self.logit_weights, dim=0)

        logits = (
            weights[0] * cnn_logits +
            weights[1] * vit_logits +
            weights[2] * artifact_logits
        )

        if return_gates:
            gates = weights.view(1, 3).repeat(x.size(0), 1)
            return logits, gates

        return logits
