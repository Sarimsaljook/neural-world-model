from __future__ import annotations

import torch
import torch.nn.functional as F

def focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = p * targets + (1.0 - p) * (1.0 - targets)
    w = (1.0 - pt) ** gamma
    return (w * ce).mean()
