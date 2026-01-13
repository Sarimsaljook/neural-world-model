from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RetrievalConfig:
    topk: int = 8


class MemoryRetrieval:
    def __init__(self, cfg: Optional[RetrievalConfig] = None):
        self.cfg = cfg or RetrievalConfig()

    def retrieve_episodic_by_embedding(self, episodes: List[Dict[str, Any]], emb: torch.Tensor, topk: Optional[int] = None) -> List[Tuple[int, float]]:
        topk = int(topk or self.cfg.topk)
        if not episodes:
            return []
        emb = F.normalize(emb.float(), dim=-1)
        sims = []
        for i, ep in enumerate(episodes):
            e = ep.get("erfg", {})
            ents = e.get("entities", [])
            if not ents:
                continue
            cand = []
            for ent in ents:
                v = ent.get("emb", None)
                if v is None:
                    continue
                cand.append(torch.as_tensor(v, dtype=torch.float32))
            if not cand:
                continue
            M = F.normalize(torch.stack(cand, dim=0), dim=-1)
            s = float((M @ emb).max().item())
            sims.append((i, s))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:topk]
