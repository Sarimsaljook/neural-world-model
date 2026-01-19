from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AMP:
    enabled: bool = True

    def __post_init__(self):
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.enabled)

    def autocast(self):
        return torch.cuda.amp.autocast(enabled=self.enabled)

    def backward(self, loss: torch.Tensor):
        self.scaler.scale(loss).backward()

    def step(self, opt: torch.optim.Optimizer):
        self.scaler.step(opt)
        self.scaler.update()

    def unscale_(self, opt: torch.optim.Optimizer):
        self.scaler.unscale_(opt)
