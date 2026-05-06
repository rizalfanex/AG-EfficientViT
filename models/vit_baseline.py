import timm
import torch.nn as nn


class ViTBaseline(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        model_name: str = "vit_tiny_patch16_224",
    ):
        super().__init__()

        try:
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes,
            )
        except Exception as exc:
            print(f"[Warning] Failed to load pretrained {model_name}: {exc}")
            print("[Info] Falling back to pretrained=False.")
            self.model = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=num_classes,
            )

    def forward(self, x):
        return self.model(x)
