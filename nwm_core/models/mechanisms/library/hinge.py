from __future__ import annotations

from typing import Any, Dict, Optional

import torch


class HingeConstraint:
    name = "hinge"

    @torch.no_grad()
    def forward(self, erfg: Any, meta: Dict, ctx: Any) -> Dict:
        src = meta.get("src", None)
        dst = meta.get("dst", None)
        axis = meta.get("axis", None)
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

        v = pa - pb
        if axis is None:
            axis_v = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        else:
            axis_v = torch.as_tensor(axis, dtype=torch.float32)
            axis_v = axis_v / (axis_v.norm() + 1e-6)

        proj = (v @ axis_v) * axis_v
        ortho = v - proj

        k = 0.15
        corr = -k * ortho

        return {
            "delta_pos": {src: corr, dst: -0.5 * corr},
            "delta_vel": {},
            "misc": {"kind": "hinge_constraint", "src": src, "dst": dst, "axis": axis_v.detach().cpu().tolist()},
        }

    @torch.no_grad()
    def score(self, erfg: Any, evidence_window: Dict) -> float:
        return 0.6
