from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

@dataclass
class CosineSchedule:
    warmup_steps: int
    max_steps: int
    min_lr: float

    def lr_at(self, base_lr: float, step: int) -> float:
        if step < self.warmup_steps:
            return base_lr * float(step + 1) / float(max(1, self.warmup_steps))
        t = float(step - self.warmup_steps) / float(max(1, self.max_steps - self.warmup_steps))
        t = min(1.0, max(0.0, t))
        return self.min_lr + 0.5 * (base_lr - self.min_lr) * (1.0 + math.cos(math.pi * t))
