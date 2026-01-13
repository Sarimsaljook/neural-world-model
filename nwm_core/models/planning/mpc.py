from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import torch


@dataclass(frozen=True)
class MPCConfig:
    horizon: int = 8
    iters: int = 16
    step_scale: float = 0.05
    contact_only: bool = True
    device: str = "cuda"


@dataclass(frozen=True)
class MPCAction:
    kind: str
    target_id: Optional[int]
    vec: Tuple[float, float, float]
    p: float
    cost: float


class MicroMPC:
    def __init__(self, cfg: Optional[MPCConfig] = None):
        self.cfg = cfg or MPCConfig()
        self.device = torch.device(self.cfg.device if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def propose_action(
        self,
        erfg: Any,
        constraints_score: float,
        events: List[Any],
        goal: Optional[Dict[str, Any]] = None,
    ) -> Optional[MPCAction]:
        if self.cfg.contact_only:
            has_contact = False
            for e in events:
                k = e.get("kind", "") if isinstance(e, dict) else getattr(e, "kind", "")
                if "contact" in k:
                    has_contact = True
                    break
            if not has_contact and goal is None:
                return None

        tgt = None
        if goal is not None:
            tgt = goal.get("target_id", None) or goal.get("source_id", None)

        if tgt is None:
            ents = getattr(erfg, "entities", {})
            if ents:
                tgt = next(iter(ents.keys()))

        if tgt is None:
            return None

        step = float(self.cfg.step_scale)
        vec = (0.0, 0.0, step)
        cost = float(constraints_score)
        p = float(max(0.0, 1.0 - cost))
        return MPCAction(kind="delta_pos", target_id=int(tgt), vec=vec, p=p, cost=cost)
