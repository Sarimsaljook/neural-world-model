from __future__ import annotations

import torch


def clip_grad(model: torch.nn.Module, max_norm: float) -> float:
    params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not params:
        return 0.0
    return float(torch.nn.utils.clip_grad_norm_(params, float(max_norm)))
