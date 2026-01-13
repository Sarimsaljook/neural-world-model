from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class RouterConfig:
    max_ops: int = 48
    min_p: float = 0.55
    learned: bool = False
    embed_dim: int = 768
    device: str = "cuda"


class _TinyRouter(nn.Module):
    def __init__(self, d: int, num_ops: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, num_ops),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MechanismRouter:
    OPS = ["rigid", "support", "contact", "hinge", "slider", "grasp", "contain", "deform"]

    def __init__(self, cfg: Optional[RouterConfig] = None):
        self.cfg = cfg or RouterConfig()
        self.device = torch.device(self.cfg.device if torch.cuda.is_available() else "cpu")
        self.learned = None
        if self.cfg.learned:
            self.learned = _TinyRouter(self.cfg.embed_dim, len(self.OPS)).to(self.device).eval()

    @torch.no_grad()
    def route(self, erfg: Any, evidence: Dict, relations: List[Dict], events: List[Any]) -> List[Tuple[str, Dict]]:
        active: List[Tuple[str, Dict]] = []

        for e in events:
            kind = getattr(e, "kind", "") if not isinstance(e, dict) else e.get("kind", "")
            src = getattr(e, "src", None) if not isinstance(e, dict) else e.get("src", None)
            dst = getattr(e, "dst", None) if not isinstance(e, dict) else e.get("dst", None)
            p = float(getattr(e, "p", 1.0) if not isinstance(e, dict) else e.get("p", 1.0))

            if p < self.cfg.min_p:
                continue

            if "contact" in kind:
                active.append(("contact", {"src": src, "dst": dst, "p": p}))
            if "grasp" in kind or "hold" in kind:
                active.append(("grasp", {"src": src, "dst": dst, "p": p}))
            if "contain" in kind or "inside" in kind:
                active.append(("contain", {"src": src, "dst": dst, "p": p}))

        for r in relations:
            k = r.get("kind", "")
            p = float(r.get("p", r.get("p_min", 1.0)))
            if p < self.cfg.min_p:
                continue
            src = r.get("src", None)
            dst = r.get("dst", None)

            if k in {"contact", "touching"}:
                active.append(("contact", {"src": src, "dst": dst, "p": p}))
            if k in {"supporting", "on", "supports"}:
                active.append(("support", {"src": src, "dst": dst, "p": p}))
            if k in {"hinge"}:
                active.append(("hinge", {"src": src, "dst": dst, "p": p, "axis": r.get("axis", None)}))
            if k in {"slider"}:
                active.append(("slider", {"src": src, "dst": dst, "p": p, "axis": r.get("axis", None)}))
            if k in {"held_by", "holding"}:
                active.append(("grasp", {"src": src, "dst": dst, "p": p}))
            if k in {"inside", "contains"}:
                active.append(("contain", {"src": src, "dst": dst, "p": p}))

        active.append(("rigid", {"p": 1.0}))

        active = self._dedupe(active)
        return active[: self.cfg.max_ops]

    def _dedupe(self, ops: List[Tuple[str, Dict]]) -> List[Tuple[str, Dict]]:
        seen = set()
        out = []
        for name, meta in ops:
            src = meta.get("src", None)
            dst = meta.get("dst", None)
            key = (name, src, dst)
            if key in seen:
                continue
            seen.add(key)
            out.append((name, meta))
        return out
