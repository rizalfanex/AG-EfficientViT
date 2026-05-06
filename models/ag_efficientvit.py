import timm
import torch.nn as nn

from models.artifact_branch import ArtifactBranch
from models.fusion import AdaptiveGatedFusion


class AGEfficientViT(nn.Module):
    """
    Artifact Guided EfficientViT.

    Architecture:
        1. EfficientNetB0 CNN branch for local spatial and texture features.
        2. ViT-Tiny branch for global contextual representation.
        3. Artifact branch for high-pass residual synthetic traces.
        4. Adaptive gated fusion for dynamic feature weighting.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        cnn_name: str = "efficientnet_b0",
        vit_name: str = "vit_tiny_patch16_224",
        artifact_dim: int = 128,
        fusion_dim: int = 256,
        dropout: float = 0.35,
    ):
        super().__init__()

        self.cnn = self._create_backbone(cnn_name, pretrained)
        self.vit = self._create_backbone(vit_name, pretrained)

        self.cnn_dim = self.cnn.num_features
        self.vit_dim = self.vit.num_features
        self.artifact_dim = artifact_dim

        self.artifact_branch = ArtifactBranch(artifact_dim=artifact_dim)

        self.fusion = AdaptiveGatedFusion(
            cnn_dim=self.cnn_dim,
            vit_dim=self.vit_dim,
            artifact_dim=artifact_dim,
            hidden_dim=fusion_dim,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
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
