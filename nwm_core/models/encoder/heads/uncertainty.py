from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyHead(nn.Module):
    def __init__(self, feat_dim: int):
        super().__init__()
        self.pixel = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(feat_dim // 2, 1, 1),
        )
        self.instance = nn.Linear(feat_dim, 1)

    def forward(self, feat_map: torch.Tensor, inst: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        px_u = F.softplus(self.pixel(feat_map)).squeeze(1)

        inst_u = None
        if inst is not None:
            embs = None
            if isinstance(inst, dict):
                if "embeddings" in inst and torch.is_tensor(inst["embeddings"]):
                    embs = inst["embeddings"]
                elif "instance_feats" in inst and torch.is_tensor(inst["instance_feats"]):
                    embs = inst["instance_feats"]

            if embs is not None:
                inst_u = F.softplus(self.instance(embs)).squeeze(-1)

        return {"pixel": px_u, "instance": inst_u}
