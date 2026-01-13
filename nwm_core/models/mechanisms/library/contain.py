from __future__ import annotations

from typing import Any, Dict

import torch


class ContainmentPouring:
    name = "contain"

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

        pa = getattr(a, "pos", None) if not isinstance(a, dict) else a.get("pos", None)
        pb = getattr(b, "pos", None) if not isinstance(b, dict) else b.get("pos", None)
        if pa is None or pb is None:
            return {"delta_pos": {}, "delta_vel": {}, "misc": None}

        pa = torch.as_tensor(pa, dtype=torch.float32)
        pb = torch.as_tensor(pb, dtype=torch.float32)
        d = (pa - pb).norm().item()

        return {
            "delta_pos": {},
            "delta_vel": {},
            "misc": {"kind": "containment_eval", "src": src, "dst": dst, "center_dist": float(d)},
        }

    @torch.no_grad()
    def score(self, erfg: Any, evidence_window: Dict) -> float:
        return 0.55
