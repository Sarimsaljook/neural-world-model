from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import torch

@dataclass
class EMA:
    decay: float = 0.999

    def __post_init__(self) -> None:
        self._shadow = {}

    def init(self, model: torch.nn.Module) -> None:
        self._shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        if not self._shadow:
            self.init(model)
        for k, v in model.state_dict().items():
            self._shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def apply_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self._shadow, strict=False)
