import timm
import torch
import torch.nn as nn


class EfficientViTBaseline(nn.Module):
    """
    Baseline hybrid model:
    EfficientNetB0 branch + ViT-Tiny branch + simple concatenation classifier.

    This is the direct hybrid baseline before adding:
    1. artifact branch
    2. adaptive gated fusion
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        cnn_name: str = "efficientnet_b0",
        vit_name: str = "vit_tiny_patch16_224",
        dropout: float = 0.3,
    ):
        super().__init__()

        self.cnn = self._create_backbone(cnn_name, pretrained)
        self.vit = self._create_backbone(vit_name, pretrained)

        self.cnn_dim = self.cnn.num_features
        self.vit_dim = self.vit.num_features
        fusion_dim = self.cnn_dim + self.vit_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
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

    def forward(self, x):
        cnn_feat = self.cnn(x)
        vit_feat = self.vit(x)

        fused = torch.cat([cnn_feat, vit_feat], dim=1)
        logits = self.classifier(fused)

        return logits
