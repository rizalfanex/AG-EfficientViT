import torch
import torch.nn as nn


class GatedConcatFusion(nn.Module):
    """
    Gated concatenation fusion.

    Difference from V1:
    - V1 projects all branches to the same dimension and sums them.
    - V2 keeps branch-specific representations and concatenates them after gating.

    This reduces information loss from CNN, ViT, and artifact branches.
    """

    def __init__(
        self,
        cnn_dim: int,
        vit_dim: int,
        artifact_dim: int,
        cnn_proj_dim: int = 512,
        vit_proj_dim: int = 256,
        artifact_proj_dim: int = 128,
        gate_hidden_dim: int = 256,
        dropout: float = 0.35,
    ):
        super().__init__()

        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_dim, cnn_proj_dim),
            nn.LayerNorm(cnn_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.vit_proj = nn.Sequential(
            nn.Linear(vit_dim, vit_proj_dim),
            nn.LayerNorm(vit_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.artifact_proj = nn.Sequential(
            nn.Linear(artifact_dim, artifact_proj_dim),
            nn.LayerNorm(artifact_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        gate_input_dim = cnn_dim + vit_dim + artifact_dim

        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, gate_hidden_dim),
            nn.LayerNorm(gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, 3),
            nn.Softmax(dim=1),
        )

        self.out_dim = cnn_proj_dim + vit_proj_dim + artifact_proj_dim

    def forward(self, cnn_feat, vit_feat, artifact_feat):
        raw = torch.cat([cnn_feat, vit_feat, artifact_feat], dim=1)
        gates = self.gate(raw)

        cnn_p = self.cnn_proj(cnn_feat) * gates[:, 0:1]
        vit_p = self.vit_proj(vit_feat) * gates[:, 1:2]
        artifact_p = self.artifact_proj(artifact_feat) * gates[:, 2:3]

        fused = torch.cat([cnn_p, vit_p, artifact_p], dim=1)

        return fused, gates
