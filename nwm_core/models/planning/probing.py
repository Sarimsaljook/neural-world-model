from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
from ...models.erfg.state import ERFGState


@dataclass(frozen=True)
class ProbeAction:
    kind: str
    target: Optional[str]
    score: float
    reason: str


@dataclass(frozen=True)
class ProbingConfig:
    uncertainty_hi: float = 18.0
    max_actions: int = 3


def propose_probes(
    state: ERFGState,
    entity_uncertainty: Dict[str, float],
    intuition: Dict[str, Dict[str, float]],
    cfg: ProbingConfig,
) -> List[ProbeAction]:
    candidates: List[ProbeAction] = []

    for eid, u in sorted(entity_uncertainty.items(), key=lambda kv: -kv[1]):
        if u < cfg.uncertainty_hi:
            continue
        risk = float(intuition.get(eid, {}).get("stability_risk", 0.0))
        reason = f"high uncertainty ({u:.1f})" + (f", risk={risk:.2f}" if risk > 0 else "")
        candidates.append(ProbeAction(kind="move_closer", target=eid, score=float(u), reason=reason))
        candidates.append(ProbeAction(kind="change_viewpoint", target=eid, score=float(u * 0.9), reason="disambiguate geometry/occlusion"))
        candidates.append(ProbeAction(kind="pan_left", target=eid, score=float(u * 0.6), reason="break occlusion"))
        candidates.append(ProbeAction(kind="pan_right", target=eid, score=float(u * 0.6), reason="break occlusion"))

    if not candidates:
        return [ProbeAction(kind="wait", target=None, score=0.0, reason="uncertainty below threshold")]

    candidates.sort(key=lambda a: -a.score)
    return candidates[: cfg.max_actions]
