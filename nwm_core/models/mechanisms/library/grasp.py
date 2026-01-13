from __future__ import annotations

from typing import Any, Dict

import torch


class GraspHoldRelease:
    name = "grasp"

    @torch.no_grad()
    def forward(self, erfg: Any, meta: Dict, ctx: Any) -> Dict:
        src = meta.get("src", None)
        dst = meta.get("dst", None)
        if src is None or dst is None:
            return {"delta_pos": {}, "delta_vel": {}, "misc": None}

        ents = getattr(erfg, "entities", {})
        obj = ents.get(dst, None)
        hand = ents.get(src, None)
        if obj is None or hand is None:
            return {"delta_pos": {}, "delta_vel": {}, "misc": None}

        ph = getattr(hand, "pos", None) if not isinstance(hand, dict) else hand.get("pos", None)
        po = getattr(obj, "pos", None) if not isinstance(obj, dict) else obj.get("pos", None)
        if ph is None or po is None:
            return {"delta_pos": {}, "delta_vel": {}, "misc": None}

        ph = torch.as_tensor(ph, dtype=torch.float32)
        po = torch.as_tensor(po, dtype=torch.float32)

        k = 0.35
        dp = k * (ph - po)

        return {
            "delta_pos": {dst: dp},
            "delta_vel": {},
            "misc": {"kind": "grasp_pull", "hand": src, "obj": dst},
        }

    @torch.no_grad()
    def score(self, erfg: Any, evidence_window: Dict) -> float:
        return 0.7
