from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Any, Sequence

import torch
import torch.nn as nn


@dataclass(frozen=True)
class DistillConfig:
    max_items: int = 50000
    embed_dim: int = 256
    device: str = "cuda"


class DistillationBuffer:
    def __init__(self, cfg: Optional[DistillConfig] = None):
        self.cfg = cfg or DistillConfig()
        self.x: List[torch.Tensor] = []
        self.y: List[torch.Tensor] = []

    def add(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x.append(x.detach().cpu())
        self.y.append(y.detach().cpu())
        if len(self.x) > self.cfg.max_items:
            self.x = self.x[-self.cfg.max_items :]
            self.y = self.y[-self.cfg.max_items :]

    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        n = min(n, len(self.x))
        if n <= 0:
            return torch.empty(0), torch.empty(0)
        idx = torch.randint(0, len(self.x), (n,))
        X = torch.stack([self.x[i] for i in idx.tolist()], dim=0)
        Y = torch.stack([self.y[i] for i in idx.tolist()], dim=0)
        return X, Y


class DistillationPolicyHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class PolicyDistillConfig:
    in_dim: int = 256
    out_dim: int = 8
    lr: float = 1e-3
    batch_size: int = 64
    warmup: int = 256
    train_steps_per_call: int = 1
    max_items: int = 50000
    device: str = "cuda"


class PolicyDistill:

    def __init__(self, cfg: Optional[PolicyDistillConfig] = None):
        self.cfg = cfg or PolicyDistillConfig()
        self.device = torch.device(self.cfg.device if torch.cuda.is_available() else "cpu")

        self.buffer = DistillationBuffer(
            DistillConfig(max_items=self.cfg.max_items, embed_dim=self.cfg.in_dim, device=str(self.device))
        )

        self.head = DistillationPolicyHead(self.cfg.in_dim, self.cfg.out_dim).to(self.device)
        self.opt = torch.optim.AdamW(self.head.parameters(), lr=self.cfg.lr)
        self.loss_fn = nn.SmoothL1Loss()

        self._seen = 0

    @torch.no_grad()
    def _default_embed(self, erfg: Any, events: Optional[Sequence[Any]], intuition: Optional[dict], goal: Optional[dict]) -> torch.Tensor:
        x = torch.zeros(self.cfg.in_dim, dtype=torch.float32)

        # basic signals
        n_ent = len(getattr(erfg, "entities", [])) if erfg is not None else 0
        n_evt = len(events) if events is not None else 0
        x[0] = float(n_ent)
        x[1] = float(n_evt)

        # intuition scalars
        if isinstance(intuition, dict):
            stab = intuition.get("stability", {})
            slip = intuition.get("slip", {})
            if isinstance(stab, dict) and len(stab) > 0:
                x[2] = float(sum(stab.values()) / max(1, len(stab)))
            if isinstance(slip, dict) and len(slip) > 0:
                x[3] = float(sum(slip.values()) / max(1, len(slip)))

        # goal scalars
        if isinstance(goal, dict):
            x[4] = float(goal.get("priority", 0))
            x[5] = float(goal.get("horizon_s", 0.0))

        return x.unsqueeze(0)  # (1, D)

    def add_pair(self, state_embed: torch.Tensor, action_target: torch.Tensor) -> None:
        if state_embed.ndim == 1:
            state_embed = state_embed[None, :]
        if action_target.ndim == 1:
            action_target = action_target[None, :]

        state_embed = state_embed.to(self.device, dtype=torch.float32)
        action_target = action_target.to(self.device, dtype=torch.float32)

        self.buffer.add(state_embed.squeeze(0), action_target.squeeze(0))
        self._seen += 1

    def predict(self, state_embed: torch.Tensor) -> torch.Tensor:
        if state_embed.ndim == 1:
            state_embed = state_embed[None, :]
        return self.head(state_embed.to(self.device, dtype=torch.float32))

    def train_step(self) -> Optional[float]:
        if len(self.buffer.x) < max(self.cfg.warmup, self.cfg.batch_size):
            return None

        X, Y = self.buffer.sample(self.cfg.batch_size)
        X = X.to(self.device, dtype=torch.float32)
        Y = Y.to(self.device, dtype=torch.float32)

        pred = self.head(X)
        loss = self.loss_fn(pred, Y)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return float(loss.item())

    def step(self, erfg: Any, events=None, intuition=None, goal=None, state_embed: Optional[torch.Tensor] = None, action_target: Optional[torch.Tensor] = None) -> dict:
        if state_embed is None:
            state_embed = self._default_embed(erfg, events, intuition, goal)

        out: dict = {"enabled": True}

        if action_target is not None:
            self.add_pair(state_embed, action_target)
            loss = None
            for _ in range(int(self.cfg.train_steps_per_call)):
                loss = self.train_step()
            out["loss"] = loss
            out["buffer_size"] = len(self.buffer.x)

        out["pred"] = self.predict(state_embed).detach()
        return out

