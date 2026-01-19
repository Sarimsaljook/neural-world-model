from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


def l1(x: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    d = (x - y).abs()
    if mask is not None:
        d = d * mask
        return d.sum() / (mask.sum().clamp(min=1.0))
    return d.mean()


def huber(x: torch.Tensor, y: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    return F.smooth_l1_loss(x, y, beta=float(delta))


def cosine(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return 1.0 - (x * y).sum(dim=-1).mean()


def bce_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets)


def ce_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, targets)
