from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass(frozen=True)
class ExecutorConfig:
    max_active_ops: int = 48
    delta_clip_pos_m: float = 0.25
    delta_clip_vel_mps: float = 2.0
    device: str = "cuda"


@dataclass(frozen=True)
class MechanismBatchContext:
    dt: float
    evidence: Dict[str, Any]
    relations: List[Dict[str, Any]]
    events: List[Any]


def _get_entity_attr(ent: Any, key: str, default=None):
    if hasattr(ent, key):
        return getattr(ent, key)
    if isinstance(ent, dict):
        return ent.get(key, default)
    return default


def _set_entity_attr(ent: Any, key: str, value):
    if hasattr(ent, key):
        setattr(ent, key, value)
    elif isinstance(ent, dict):
        ent[key] = value


class MechanismExecutor:
    def __init__(self, cfg: Optional[ExecutorConfig] = None):
        self.cfg = cfg or ExecutorConfig()

    @torch.no_grad()
    def step(
        self,
        erfg: Any,
        active_ops: List[Tuple[str, Dict[str, Any]]],
        ctx: MechanismBatchContext,
        library: Dict[str, Any],
    ) -> Dict[str, Any]:
        device = torch.device(self.cfg.device if torch.cuda.is_available() else "cpu")

        applied: List[Dict[str, Any]] = []
        delta_pos = {}
        delta_vel = {}
        delta_misc: List[Dict[str, Any]] = []

        for name, meta in active_ops[: self.cfg.max_active_ops]:
            op = library.get(name, None)
            if op is None:
                continue
            out = op.forward(erfg, meta, ctx)
            applied.append({"name": name, "meta": meta, "out": out})

            dpos = out.get("delta_pos", None)
            dvel = out.get("delta_vel", None)
            misc = out.get("misc", None)

            if isinstance(dpos, dict):
                for k, v in dpos.items():
                    if v is None:
                        continue
                    v = torch.as_tensor(v, device=device, dtype=torch.float32)
                    if k not in delta_pos:
                        delta_pos[k] = v
                    else:
                        delta_pos[k] = delta_pos[k] + v

            if isinstance(dvel, dict):
                for k, v in dvel.items():
                    if v is None:
                        continue
                    v = torch.as_tensor(v, device=device, dtype=torch.float32)
                    if k not in delta_vel:
                        delta_vel[k] = v
                    else:
                        delta_vel[k] = delta_vel[k] + v

            if misc:
                delta_misc.append(misc)

        self._apply_deltas(erfg, delta_pos, delta_vel, ctx.dt)
        return {"applied": applied, "delta_misc": delta_misc}

    def _apply_deltas(self, erfg: Any, dpos: Dict[int, torch.Tensor], dvel: Dict[int, torch.Tensor], dt: float) -> None:
        ents = getattr(erfg, "entities", None)
        if ents is None:
            return

        for eid, ent in list(ents.items()):
            if eid in dvel:
                dv = dvel[eid].clamp(-self.cfg.delta_clip_vel_mps, self.cfg.delta_clip_vel_mps)
                v = _get_entity_attr(ent, "vel", None)
                if v is None:
                    _set_entity_attr(ent, "vel", dv.detach().cpu())
                else:
                    v_t = torch.as_tensor(v, dtype=torch.float32)
                    v_t = (v_t + dv.detach().cpu()).detach()
                    _set_entity_attr(ent, "vel", v_t)

            if eid in dpos:
                dp = dpos[eid].clamp(-self.cfg.delta_clip_pos_m, self.cfg.delta_clip_pos_m)
                p = _get_entity_attr(ent, "pos", None)
                if p is None:
                    _set_entity_attr(ent, "pos", dp.detach().cpu())
                else:
                    p_t = torch.as_tensor(p, dtype=torch.float32)
                    p_t = (p_t + dp.detach().cpu()).detach()
                    _set_entity_attr(ent, "pos", p_t)

        for eid, ent in list(ents.items()):
            v = _get_entity_attr(ent, "vel", None)
            p = _get_entity_attr(ent, "pos", None)
            if v is None or p is None:
                continue
            v_t = torch.as_tensor(v, dtype=torch.float32)
            p_t = torch.as_tensor(p, dtype=torch.float32)
            p_new = (p_t + v_t * float(dt)).detach()
            _set_entity_attr(ent, "pos", p_new)
