from __future__ import annotations

from typing import Any, Dict

import torch


class DeformationBend:
    name = "deform"

    @torch.no_grad()
    def forward(self, erfg: Any, meta: Dict, ctx: Any) -> Dict:
        src = meta.get("src", None)
        if src is None:
            return {"delta_pos": {}, "delta_vel": {}, "misc": None}
        return {"delta_pos": {}, "delta_vel": {}, "misc": {"kind": "deform_stub", "src": src}}

    @torch.no_grad()
    def score(self, erfg: Any, evidence_window: Dict) -> float:
        return 0.25
