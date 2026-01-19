from __future__ import annotations

import math
from dataclasses import dataclass

@dataclass
class WarmupCosine:
    base_lr: float
    warmup_steps: int
    total_steps: int
    min_lr: float = 0.0

    def lr(self, step: int) -> float:
        step = int(step)
        if step < self.warmup_steps:
            return self.base_lr * (step + 1) / max(1, self.warmup_steps)
        t = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        t = min(max(t, 0.0), 1.0)
        c = 0.5 * (1.0 + math.cos(math.pi * t))
        return self.min_lr + (self.base_lr - self.min_lr) * c
