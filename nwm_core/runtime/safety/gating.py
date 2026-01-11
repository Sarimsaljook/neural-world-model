from __future__ import annotations

from dataclasses import dataclass

@dataclass
class SafetyGate:
    max_risk: float = 0.7
    min_confidence: float = 0.5

    def allow(self, risk: float, confidence: float) -> bool:
        return float(risk) <= self.max_risk and float(confidence) >= self.min_confidence
