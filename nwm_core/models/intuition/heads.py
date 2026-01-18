from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class IntuitionHeadsConfig:
    in_dim: int = 256
    hidden: int = 256
    dropout: float = 0.0

    out_stability: bool = True
    out_slip: bool = True
    out_support: bool = True
    out_collision: bool = True


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = self.drop(x)
        x = F.gelu(self.fc2(x))
        x = self.drop(x)
        return self.fc3(x)


class IntuitionHeads(nn.Module):

    def __init__(self, cfg: Optional[IntuitionHeadsConfig] = None):
        super().__init__()
        self.cfg = cfg or IntuitionHeadsConfig()

        self.stability = _MLP(self.cfg.in_dim, self.cfg.hidden, 1, self.cfg.dropout) if self.cfg.out_stability else None
        self.slip = _MLP(self.cfg.in_dim, self.cfg.hidden, 1, self.cfg.dropout) if self.cfg.out_slip else None
        self.support = _MLP(self.cfg.in_dim, self.cfg.hidden, 1, self.cfg.dropout) if self.cfg.out_support else None

        self.collision = (
            _MLP(self.cfg.in_dim * 2, self.cfg.hidden, 1, self.cfg.dropout) if self.cfg.out_collision else None
        )

    def forward(
        self,
        ent_feats: torch.Tensor,
        pair_feats: Optional[torch.Tensor] = None,
        pair_index: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if ent_feats.dim() == 2:
            ent = ent_feats
        elif ent_feats.dim() == 3:
            ent = ent_feats.view(-1, ent_feats.shape[-1])
        else:
            raise ValueError(f"ent_feats must be (N,F) or (B,N,F), got {tuple(ent_feats.shape)}")

        out: Dict[str, torch.Tensor] = {}

        if self.stability is not None:
            out["stability_logits"] = self.stability(ent).squeeze(-1)

        if self.slip is not None:
            out["slip_logits"] = self.slip(ent).squeeze(-1)

        if self.support is not None:
            out["support_logits"] = self.support(ent).squeeze(-1)

        if self.collision is not None:
            if pair_feats is not None:
                pf = pair_feats
            else:
                if pair_index is None:
                    raise ValueError("collision head needs pair_feats or pair_index")
                i = pair_index[:, 0].long()
                j = pair_index[:, 1].long()
                pf = torch.cat([ent[i], ent[j]], dim=-1)

            out["collision_logits"] = self.collision(pf).squeeze(-1)

        return out
