from __future__ import annotations

from typing import Any, Dict

import torch


class ContactImpulse:
    name = "contact"

    @torch.no_grad()
    def forward(self, erfg: Any, meta: Dict, ctx: Any) -> Dict:
        src = meta.get("src", None)
        dst = meta.get("dst", None)
        if src is None or dst is None:
            return {"delta_pos": {}, "delta_vel": {}, "misc": None}

        ents = getattr(erfg, "entities", {})
        a = ents.get(src, None)
        b = ents.get(dst, None)
        if a is None or b is None:
            return {"delta_pos": {}, "delta_vel": {}, "misc": None}

        va = getattr(a, "vel", None) if not isinstance(a, dict) else a.get("vel", None)
        vb = getattr(b, "vel", None) if not isinstance(b, dict) else b.get("vel", None)
        if va is None or vb is None:
            return {"delta_pos": {}, "delta_vel": {}, "misc": None}

        va = torch.as_tensor(va, dtype=torch.float32)
        vb = torch.as_tensor(vb, dtype=torch.float32)
        rel = va - vb

        damp = 0.35
        dv = -damp * rel

        return {
            "delta_pos": {},
            "delta_vel": {src: dv, dst: -0.5 * dv},
            "misc": {"kind": "contact_impulse", "src": src, "dst": dst, "damp": damp},
        }

    @torch.no_grad()
    def score(self, erfg: Any, evidence_window: Dict) -> float:
        return 0.65
