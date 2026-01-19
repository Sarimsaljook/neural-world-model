from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import torch

from nwm_core.models.mechanisms.router import MechanismRouter
from nwm_core.models.mechanisms.executor import MechanismExecutor
from nwm_core.models.mechanisms.losses import MechanismLosses, MechanismLossConfig

from nwm_core.train.optim import AMP, WarmupCosine, clip_grad
from nwm_core.train.stages._data import build_datapipe


def _frame(video: torch.Tensor, t: int) -> torch.Tensor:
    return video[:, t]


class Stage3Mechanisms:
    def __init__(self, device, data_cfg, stage_cfg, workdir: Path, logger, ckpt):
        self.device = device
        self.data_cfg = data_cfg
        self.cfg = stage_cfg
        self.workdir = workdir
        self.logger = logger
        self.ckpt = ckpt

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        encoder = state["encoder"].to(self.device).eval()
        compiler = state["compiler"]

        mech_router = MechanismRouter()
        mech_exec = MechanismExecutor()

        losses = MechanismLosses(MechanismLossConfig())

        params = []
        for m in [mech_router, mech_exec, losses]:
            if hasattr(m, "parameters"):
                params += [p for p in m.parameters() if p.requires_grad]

        pipe = build_datapipe(self.data_cfg, self.cfg)
        it = pipe.train_iter()

        lr = float(self.cfg.get("lr", 2e-4))
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=float(self.cfg.get("weight_decay", 0.02))) if params else None
        sched = WarmupCosine(base_lr=lr, warmup_steps=int(self.cfg.get("warmup_steps", 1000)), total_steps=int(self.cfg.get("total_steps", 15000)), min_lr=float(self.cfg.get("min_lr", 1e-6)))
        amp = AMP(enabled=bool(self.cfg.get("amp", True)))

        step = 0
        total_steps = int(self.cfg.get("total_steps", 15000))
        t0 = time.time()

        while step < total_steps:
            batch = next(it)
            video = batch["video"].to(self.device, non_blocking=True)

            x0 = _frame(video, 0)
            x1 = _frame(video, 1)

            with torch.no_grad():
                ev0 = encoder(x0, None)
                ev1 = encoder(x1, x0)

            w0 = compiler.step(ev0, dt=1.0 / 15.0)
            w1 = compiler.step(ev1, dt=1.0 / 15.0)

            erfg = w1["erfg"]
            events = w1.get("events", [])

            with amp.autocast():
                active = mech_router.route(erfg, events, ev1)
                mres = mech_exec.step(erfg, active, dt=1.0 / 15.0, evidence=ev1)
                erfg2 = mres.get("erfg", erfg)
                events2 = mres.get("events", [])
                loss = losses(erfg, events, active, erfg2, events2)

            if opt is not None:
                opt.zero_grad(set_to_none=True)
                amp.backward(loss)
                amp.unscale_(opt)
                gn = clip_grad(mech_exec, float(self.cfg.get("grad_clip", 1.0)))
                for pg in opt.param_groups:
                    pg["lr"] = sched.lr(step)
                amp.step(opt)
            else:
                gn = 0.0

            if step % int(self.cfg.get("log_every", 50)) == 0:
                self.logger.log({"stage": "stage3_mechanisms", "step": step, "loss": float(loss.item()), "grad_norm": float(gn), "sec": time.time() - t0})

            if step % int(self.cfg.get("ckpt_every", 2000)) == 0 and step > 0:
                payload = {"step": step}
                if hasattr(mech_router, "state_dict"):
                    payload["mech_router"] = mech_router.state_dict()
                if hasattr(mech_exec, "state_dict"):
                    payload["mech_exec"] = mech_exec.state_dict()
                self.ckpt.save(f"stage3_mechanisms_step{step:07d}", payload)

            step += 1

        state["mech_router"] = mech_router
        state["mech_exec"] = mech_exec
        return state
