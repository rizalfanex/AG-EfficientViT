import timm
import torch.nn as nn

from models.artifact_branch import ArtifactBranch
from models.fusion_v2 import GatedConcatFusion


class AGEfficientViTV2(nn.Module):
    """
    Artifact Guided EfficientViT V2.

    Main improvement over V1:
        V1 uses weighted-sum fusion.
        V2 uses gated concatenation fusion to preserve branch-specific information.

    Branches:
        1. EfficientNetB0 CNN branch
        2. ViT-Tiny Transformer branch
        3. High-pass artifact branch
        4. Gated concatenation fusion
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        cnn_name: str = "efficientnet_b0",
        vit_name: str = "vit_tiny_patch16_224",
        artifact_dim: int = 128,
        dropout: float = 0.35,
    ):
        super().__init__()

        self.cnn = self._create_backbone(cnn_name, pretrained)
        self.vit = self._create_backbone(vit_name, pretrained)

        self.cnn_dim = self.cnn.num_features
        self.vit_dim = self.vit.num_features
        self.artifact_dim = artifact_dim

        self.artifact_branch = ArtifactBranch(artifact_dim=artifact_dim)

        self.fusion = GatedConcatFusion(
            cnn_dim=self.cnn_dim,
            vit_dim=self.vit_dim,
            artifact_dim=artifact_dim,
            cnn_proj_dim=512,
            vit_proj_dim=256,
            artifact_proj_dim=128,
            gate_hidden_dim=256,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.fusion.out_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(128, num_classes),
        )

    def _create_backbone(self, model_name: str, pretrained: bool):
        try:
            model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,
                global_pool="avg",
            )
        except Exception as exc:
            print(f"[Warning] Failed to load pretrained {model_name}: {exc}")
            print("[Info] Falling back to pretrained=False.")
            model = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=0,
                global_pool="avg",
            )

        return model

    def forward(self, x, return_gates: bool = False):
        cnn_feat = self.cnn(x)
        vit_feat = self.vit(x)
        artifact_feat = self.artifact_branch(x)

        fused, gates = self.fusion(cnn_feat, vit_feat, artifact_feat)
        logits = self.classifier(fused)

        if return_gates:
            return logits, gates

        return logits
