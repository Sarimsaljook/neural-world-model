from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class RuleMemoryConfig:
    max_rules: int = 2048
    ema_beta: float = 0.97


class RuleMemory:
    def __init__(self, cfg: Optional[RuleMemoryConfig] = None):
        self.cfg = cfg or RuleMemoryConfig()
        self.rules: Dict[str, Dict[str, Any]] = {}

    def update(self, key: str, params: Dict[str, float]) -> None:
        if key not in self.rules:
            self.rules[key] = {"n": 1, "params": dict(params)}
            self._cap()
            return

        beta = float(self.cfg.ema_beta)
        cur = self.rules[key]["params"]
        for k, v in params.items():
            v = float(v)
            if k not in cur:
                cur[k] = v
            else:
                cur[k] = beta * float(cur[k]) + (1.0 - beta) * v
        self.rules[key]["n"] = int(self.rules[key]["n"]) + 1
        self._cap()

    def get(self, key: str) -> Optional[Dict[str, float]]:
        r = self.rules.get(key, None)
        if r is None:
            return None
        return dict(r["params"])

    def _cap(self) -> None:
        if len(self.rules) <= self.cfg.max_rules:
            return
        items = sorted(self.rules.items(), key=lambda kv: int(kv[1].get("n", 0)), reverse=True)
        self.rules = dict(items[: self.cfg.max_rules])
