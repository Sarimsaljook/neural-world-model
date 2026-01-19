from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class EMA:
    decay: float = 0.999

    def __post_init__(self):
        self.shadow: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def init(self, model: torch.nn.Module):
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        sd = model.state_dict()
        if not self.shadow:
            self.init(model)
            return
        d = float(self.decay)
        for k, v in sd.items():
            if k not in self.shadow:
                self.shadow[k] = v.detach().clone()
            else:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=(1.0 - d))

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module):
        if not self.shadow:
            return
        model.load_state_dict(self.shadow, strict=False)
