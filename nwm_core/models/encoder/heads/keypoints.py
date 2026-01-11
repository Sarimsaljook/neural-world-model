from __future__ import annotations
import torch
from torch import nn

class KeypointsHead(nn.Module):
    def __init__(self, in_dim: int, k: int = 21) -> None:
        super().__init__()
        self.out = nn.Conv2d(in_dim, 2 * k, 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.out(feat)
