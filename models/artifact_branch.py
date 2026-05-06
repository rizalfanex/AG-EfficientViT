import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedHighPassLayer(nn.Module):
    """
    Fixed high-pass filtering layer using a Laplacian kernel.
    This layer emphasizes residual artifact patterns such as abnormal edges,
    local noise, texture inconsistency, and synthetic image traces.
    """

    def __init__(self, channels: int = 3):
        super().__init__()

        kernel = torch.tensor(
            [[0.0, -1.0, 0.0],
             [-1.0, 4.0, -1.0],
             [0.0, -1.0, 0.0]],
            dtype=torch.float32,
        )

        weight = kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer("weight", weight)
        self.channels = channels

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=1, groups=self.channels)


class ArtifactBranch(nn.Module):
    """
    Lightweight artifact feature extractor.

    Input:
        Image tensor [B, 3, H, W]

    Output:
        Artifact feature vector [B, artifact_dim]
    """

    def __init__(self, artifact_dim: int = 128):
        super().__init__()

        self.high_pass = FixedHighPassLayer(channels=3)

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, artifact_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(artifact_dim),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.norm = nn.LayerNorm(artifact_dim)

    def forward(self, x):
        residual = self.high_pass(x)
        feat = self.encoder(residual)
        feat = self.norm(feat)
        return feat
