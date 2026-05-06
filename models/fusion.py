import torch
import torch.nn as nn


class AdaptiveGatedFusion(nn.Module):
    """
    Adaptive gated fusion for combining CNN, Transformer, and artifact features.

    Instead of simple concatenation, this module learns how much each branch
    should contribute for each input image.
    """

    def __init__(
        self,
        cnn_dim: int,
        vit_dim: int,
        artifact_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.vit_proj = nn.Sequential(
            nn.Linear(vit_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.artifact_proj = nn.Sequential(
            nn.Linear(artifact_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        gate_input_dim = cnn_dim + vit_dim + artifact_dim

        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=1),
        )

    def forward(self, cnn_feat, vit_feat, artifact_feat):
        cnn_p = self.cnn_proj(cnn_feat)
        vit_p = self.vit_proj(vit_feat)
        artifact_p = self.artifact_proj(artifact_feat)

        raw_fused = torch.cat([cnn_feat, vit_feat, artifact_feat], dim=1)
        weights = self.gate(raw_fused)

        fused = (
            weights[:, 0:1] * cnn_p +
            weights[:, 1:2] * vit_p +
            weights[:, 2:3] * artifact_p
        )

        return fused, weights
