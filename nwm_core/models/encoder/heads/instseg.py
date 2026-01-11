from __future__ import annotations
import torch
from torch import nn

class InstSegHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.cls = nn.Conv2d(in_dim, num_classes, 1)
        self.emb = nn.Conv2d(in_dim, 256, 1)

    def forward(self, feat: torch.Tensor) -> dict:
        return {"class_logits": self.cls(feat), "embeddings": self.emb(feat)}
