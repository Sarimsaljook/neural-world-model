from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SemanticConfig:
    max_concepts: int = 4096
    ema_beta: float = 0.98
    min_updates: int = 3


class SemanticMemory:
    def __init__(self, cfg: Optional[SemanticConfig] = None):
        self.cfg = cfg or SemanticConfig()
        self.concepts: Dict[str, Dict[str, Any]] = {}

    def update_from_erfg(self, erfg: Any) -> None:
        ents = getattr(erfg, "entities", {})
        for eid, e in ents.items():
            emb = getattr(e, "emb", None) if not isinstance(e, dict) else e.get("emb", None)
            cls = getattr(e, "cls", None) if not isinstance(e, dict) else e.get("cls", None)
            if emb is None:
                continue
            key = f"cls:{int(cls) if cls is not None else -1}"
            self._ema_update(key, torch.as_tensor(emb, dtype=torch.float32))

        self._cap()

    def _ema_update(self, key: str, x: torch.Tensor) -> None:
        x = F.normalize(x, dim=-1)
        if key not in self.concepts:
            self.concepts[key] = {"proto": x.clone(), "n": 1}
            return
        beta = float(self.cfg.ema_beta)
        p = self.concepts[key]["proto"]
        p = beta * p + (1.0 - beta) * x
        self.concepts[key]["proto"] = F.normalize(p, dim=-1)
        self.concepts[key]["n"] = int(self.concepts[key]["n"]) + 1

    def query(self, emb: torch.Tensor, topk: int = 5) -> List[Tuple[str, float]]:
        if not self.concepts:
            return []
        emb = F.normalize(emb.float(), dim=-1)
        keys = list(self.concepts.keys())
        protos = torch.stack([self.concepts[k]["proto"] for k in keys], dim=0)
        sim = (protos @ emb).detach()
        k = min(topk, sim.numel())
        vals, idx = torch.topk(sim, k=k, largest=True)
        out = []
        for v, i in zip(vals.tolist(), idx.tolist()):
            out.append((keys[i], float(v)))
        return out

    def _cap(self) -> None:
        if len(self.concepts) <= self.cfg.max_concepts:
            return
        items = sorted(self.concepts.items(), key=lambda kv: int(kv[1].get("n", 0)), reverse=True)
        self.concepts = dict(items[: self.cfg.max_concepts])
