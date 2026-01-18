from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass(frozen=True)
class SpatialConfig:
    grid_size: int = 96
    world_extent_m: float = 3.0
    ema_beta: float = 0.98


class SpatialMemory:
    def __init__(self, cfg: Optional[SpatialConfig] = None):
        self.cfg = cfg or SpatialConfig()
        self.heat = torch.zeros((self.cfg.grid_size, self.cfg.grid_size), dtype=torch.float32)
        self.seen = 0

    def update_from_erfg(self, erfg: Any) -> None:
        ents = getattr(erfg, "entities", {})
        if not ents:
            return
        g = self.cfg.grid_size
        ext = float(self.cfg.world_extent_m)
        beta = float(self.cfg.ema_beta)

        heat_new = torch.zeros_like(self.heat)
        for _, e in ents.items():
            pos = getattr(e, "pos", None) if not isinstance(e, dict) else e.get("pos", None)
            if pos is None:
                continue
            p = torch.as_tensor(pos, dtype=torch.float32)
            x = float(p[0].item())
            y = float(p[1].item())
            ix = int((x / ext * 0.5 + 0.5) * (g - 1))
            iy = int((y / ext * 0.5 + 0.5) * (g - 1))
            ix = max(0, min(g - 1, ix))
            iy = max(0, min(g - 1, iy))
            heat_new[iy, ix] += 1.0

        self.heat = beta * self.heat + (1.0 - beta) * heat_new
        self.seen += 1

    def get_heatmap(self) -> torch.Tensor:
        return self.heat.clone()
