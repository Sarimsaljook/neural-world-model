from __future__ import annotations

from typing import List, Tuple
import torch
import torch.nn as nn


class BeliefGNN(nn.Module):
    def __init__(self, node_dim: int, hidden: int = 256):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(node_dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, node_dim),
        )

    def forward(self, nodes: torch.Tensor, edges: List[Tuple[int, int]]) -> torch.Tensor:
        if nodes.numel() == 0 or not edges:
            return nodes
        out = nodes.clone()
        for i, j in edges:
            m_ij = self.msg(torch.cat([nodes[i], nodes[j]], dim=-1))
            m_ji = self.msg(torch.cat([nodes[j], nodes[i]], dim=-1))
            out[i] = out[i] + m_ij
            out[j] = out[j] + m_ji
        return out
