from __future__ import annotations

from dataclasses import dataclass

@dataclass
class AnomalyDetector:
    threshold: float = 3.0
    ema_decay: float = 0.99
    _ema: float = 0.0

    def update(self, surprise: float) -> bool:
        self._ema = self.ema_decay * self._ema + (1.0 - self.ema_decay) * float(surprise)
        return float(surprise) > self.threshold
