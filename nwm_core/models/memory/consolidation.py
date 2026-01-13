from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .episodic import EpisodicMemory
from .semantic import SemanticMemory
from .spatial import SpatialMemory
from .rule_memory import RuleMemory


@dataclass(frozen=True)
class ConsolidationConfig:
    episodic_stride: int = 5
    max_event_rules: int = 64


class MemoryConsolidator:
    def __init__(
        self,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        spatial: SpatialMemory,
        rules: RuleMemory,
        cfg: Optional[ConsolidationConfig] = None,
    ):
        self.cfg = cfg or ConsolidationConfig()
        self.episodic = episodic
        self.semantic = semantic
        self.spatial = spatial
        self.rules = rules
        self._step = 0

    def step(self, erfg: Any, events: List[Any], evidence: Optional[Dict] = None) -> None:
        self._step += 1

        if (self._step % self.cfg.episodic_stride) == 0:
            self.episodic.add(erfg, events, evidence=evidence)

        self.semantic.update_from_erfg(erfg)
        self.spatial.update_from_erfg(erfg)
        self._extract_rules_from_events(events)

    def _extract_rules_from_events(self, events: List[Any]) -> None:
        n = 0
        for e in events:
            kind = e.get("kind", "") if isinstance(e, dict) else getattr(e, "kind", "")
            if not kind:
                continue
            p = float(e.get("p", 1.0) if isinstance(e, dict) else getattr(e, "p", 1.0))
            if p < 0.55:
                continue
            params = e.get("params", {}) if isinstance(e, dict) else (getattr(e, "params", {}) if hasattr(e, "params") else {})
            if isinstance(params, dict) and params:
                self.rules.update(f"event:{kind}", {k: float(v) for k, v in params.items() if _is_float(v)})
                n += 1
                if n >= self.cfg.max_event_rules:
                    break


def _is_float(x) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False
