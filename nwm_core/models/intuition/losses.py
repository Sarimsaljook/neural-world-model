from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class IntuitionLossConfig:
    w_stability: float = 1.0
    w_slip: float = 1.0
    w_support: float = 1.0
    w_collision: float = 1.0

    focal_gamma: float = 0.0
    pos_weight: float = 1.0


def _bce_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: float,
    gamma: float,
) -> torch.Tensor:
    targets = targets.float()
    if gamma <= 0.0:
        pw = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)

    prob = torch.sigmoid(logits)
    bce = F.binary_cross_entropy(prob, targets, reduction="none")
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    focal = (1.0 - p_t).pow(gamma) * bce
    w = targets * pos_weight + (1.0 - targets)
    return (w * focal).mean()


class IntuitionLoss:
    def __init__(self, cfg: Optional[IntuitionLossConfig] = None):
        self.cfg = cfg or IntuitionLossConfig()

    def __call__(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        total = torch.zeros((), device=next(iter(preds.values())).device)

        if "stability_logits" in preds and "stability" in targets:
            l = _bce_logits(preds["stability_logits"], targets["stability"], self.cfg.pos_weight, self.cfg.focal_gamma)
            out["loss_stability"] = l
            total = total + self.cfg.w_stability * l

        if "slip_logits" in preds and "slip" in targets:
            l = _bce_logits(preds["slip_logits"], targets["slip"], self.cfg.pos_weight, self.cfg.focal_gamma)
            out["loss_slip"] = l
            total = total + self.cfg.w_slip * l

        if "support_logits" in preds and "support" in targets:
            l = _bce_logits(preds["support_logits"], targets["support"], self.cfg.pos_weight, self.cfg.focal_gamma)
            out["loss_support"] = l
            total = total + self.cfg.w_support * l

        if "collision_logits" in preds and "collision" in targets:
            l = _bce_logits(preds["collision_logits"], targets["collision"], self.cfg.pos_weight, self.cfg.focal_gamma)
            out["loss_collision"] = l
            total = total + self.cfg.w_collision * l

        out["loss_total"] = total
        return out
