from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math


@dataclass(frozen=True)
class ProbingConfig:
    min_uncertainty: float = 0.55
    max_probes: int = 2
    camera_only: bool = True
    yaw_deg: float = 10.0
    lateral_m: float = 0.10


@dataclass(frozen=True)
class ProbeAction:
    kind: str
    params: Dict[str, float]
    p: float
    utility: float


class ProbingPolicy:
    def __init__(self, cfg: Optional[ProbingConfig] = None):
        self.cfg = cfg or ProbingConfig()

    def propose(
        self,
        erfg: Any,
        evidence: Dict[str, Any],
        intuition: Optional[Dict[str, Any]] = None,
    ) -> List[ProbeAction]:
        intuition = intuition or {}
        out: List[ProbeAction] = []

        u_inst = evidence.get("instance_uncertainty", None)
        u_pix = evidence.get("pixel_uncertainty", None)

        u = 0.0
        if u_inst is not None:
            try:
                u = max(u, float(getattr(u_inst, "mean", lambda: u_inst)() if hasattr(u_inst, "mean") else float(u_inst)))
            except Exception:
                pass

        if u_pix is not None:
            try:
                u = max(u, float(u_pix.float().mean().item()))
            except Exception:
                pass

        if u < self.cfg.min_uncertainty:
            return out

        out.append(
            ProbeAction(
                kind="camera_yaw",
                params={"deg": float(self.cfg.yaw_deg)},
                p=min(0.95, 0.5 + 0.5 * u),
                utility=u,
            )
        )

        out.append(
            ProbeAction(
                kind="camera_lateral",
                params={"meters": float(self.cfg.lateral_m)},
                p=min(0.90, 0.4 + 0.6 * u),
                utility=u * 0.9,
            )
        )

        out.sort(key=lambda a: a.utility, reverse=True)
        return out[: self.cfg.max_probes]
