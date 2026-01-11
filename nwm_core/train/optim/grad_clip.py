from __future__ import annotations
import torch

def clip_grad_norm_(params, max_norm: float) -> float:
    return float(torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm))
