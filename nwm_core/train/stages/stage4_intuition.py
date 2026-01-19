from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import torch

from nwm_core.models.intuition.fields import IntuitionFields
from nwm_core.models.intuition.losses import IntuitionLoss, IntuitionLossConfig

from nwm_core.train.optim import AMP, WarmupCosine, clip_grad
from nwm_core.train.stages._data import build_datapipe


def _frame(video: torch.Tensor, t: int) -> torch.Tensor:
    return video[:, t]


class Stage4Intuition:
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

        intuition = IntuitionFields().to(self.device)
        losses = IntuitionLoss(IntuitionLossConfig()).to(self.device)

        params = [p for p in intuition.parameters() if p.requires_grad] + [p for p in losses.parameters() if p.requires_grad]

        pipe = build_datapipe(self.data_cfg, self.cfg)
        it = pipe.train_iter()

        lr = float(self.cfg.get("lr", 2e-4))
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=float(self.cfg.get("weight_decay", 0.02)))
        sched = WarmupCosine(base_lr=lr, warmup_steps=int(self.cfg.get("warmup_steps", 1000)), total_steps=int(self.cfg.get("total_steps", 20000)), min_lr=float(self.cfg.get("min_lr", 1e-6)))
        amp = AMP(enabled=bool(self.cfg.get("amp", True)))

        step = 0
        total_steps = int(self.cfg.get("total_steps", 20000))
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
                intu = intuition.step(erfg, dt=1.0 / 15.0, events=events, evidence=ev1)
                loss = losses(intu, events)

            opt.zero_grad(set_to_none=True)
            amp.backward(loss)
            amp.unscale_(opt)
            gn = clip_grad(intuition, float(self.cfg.get("grad_clip", 1.0)))
            for pg in opt.param_groups:
                pg["lr"] = sched.lr(step)
            amp.step(opt)

            if step % int(self.cfg.get("log_every", 50)) == 0:
                self.logger.log({"stage": "stage4_intuition", "step": step, "loss": float(loss.item()), "grad_norm": float(gn), "sec": time.time() - t0})

            if step % int(self.cfg.get("ckpt_every", 2000)) == 0 and step > 0:
                self.ckpt.save(f"stage4_intuition_step{step:07d}", {"intuition": intuition.state_dict(), "step": step})

            step += 1

        state["intuition"] = intuition.eval()
        return state
