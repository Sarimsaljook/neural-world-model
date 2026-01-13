from __future__ import annotations

from typing import Any, Dict

import torch


class RigidMotion:
    name = "rigid"

    @torch.no_grad()
    def forward(self, erfg: Any, meta: Dict, ctx: Any) -> Dict:
        ents = getattr(erfg, "entities", {})
        dvel = {}
        for eid, ent in ents.items():
            v = getattr(ent, "vel", None) if not isinstance(ent, dict) else ent.get("vel", None)
            if v is None:
                continue
            dvel[eid] = torch.zeros(3)
        return {"delta_vel": dvel, "delta_pos": {}, "misc": None}

    @torch.no_grad()
    def score(self, erfg: Any, evidence_window: Dict) -> float:
        return 1.0
