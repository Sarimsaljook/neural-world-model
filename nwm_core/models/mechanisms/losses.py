from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MechanismLossConfig:
    router_ce_w: float = 1.0
    op_nll_w: float = 0.25
    reg_w: float = 1e-4


class MechanismLosses:
    def __init__(self, cfg: Optional[MechanismLossConfig] = None):
        self.cfg = cfg or MechanismLossConfig()

    def router_ce(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, target)

    def op_nll_diag_gaussian(self, pred_mean: torch.Tensor, pred_logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logvar = pred_logvar.clamp(-10.0, 6.0)
        var = torch.exp(logvar)
        nll = 0.5 * (logvar + (target - pred_mean) ** 2 / var)
        return nll.mean()

    def l2_reg(self, params) -> torch.Tensor:
        reg = torch.zeros((), device=next(iter(params)).device)
        for p in params:
            if p is None or (not hasattr(p, "requires_grad")) or (not p.requires_grad):
                continue
            reg = reg + (p.float() ** 2).mean()
        return reg

    def total(self, losses: Dict[str, torch.Tensor], reg_params=None) -> torch.Tensor:
        total = torch.zeros((), device=next(iter(losses.values())).device)
        total = total + self.cfg.router_ce_w * losses.get("router_ce", torch.zeros_like(total))
        total = total + self.cfg.op_nll_w * losses.get("op_nll", torch.zeros_like(total))
        if reg_params is not None:
            total = total + self.cfg.reg_w * self.l2_reg(reg_params)
        return total
