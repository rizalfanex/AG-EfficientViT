import timm
import torch.nn as nn


class EfficientNetB0Baseline(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()

        try:
            self.model = timm.create_model(
                "efficientnet_b0",
                pretrained=pretrained,
                num_classes=num_classes,
            )
        except Exception as exc:
            print(f"[Warning] Failed to load pretrained EfficientNetB0: {exc}")
            print("[Info] Falling back to pretrained=False.")
            self.model = timm.create_model(
                "efficientnet_b0",
                pretrained=False,
                num_classes=num_classes,
            )

    def forward(self, x):
        return self.model(x)
