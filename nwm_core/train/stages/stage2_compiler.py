from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F

from nwm_core.train.optim import AMP, WarmupCosine, clip_grad
from nwm_core.train.stages._data import build_datapipe


def _frame(video: torch.Tensor, t: int) -> torch.Tensor:
    return video[:, t]


def _entity_tensor(erfg: Any) -> torch.Tensor:
    ents = getattr(erfg, "entities", None)
    if ents is None:
        return torch.empty(0)
    if isinstance(ents, dict):
        ents = list(ents.values())
    xs = []
    for e in ents:
        pos = getattr(e, "pose", None) or getattr(e, "pos", None)
        if pos is None and isinstance(e, dict):
            pos = e.get("pose", e.get("pos", None))
        if pos is None:
            continue
        xs.append(torch.as_tensor(pos).float().view(-1)[:3])
    if not xs:
        return torch.empty(0)
    return torch.stack(xs, dim=0)


class Stage2Compiler:
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

        pipe = build_datapipe(self.data_cfg, self.cfg)
        it = pipe.train_iter()

        # compiler may or may not have parameters; support both
        params = []
        if hasattr(compiler, "parameters"):
            params = [p for p in compiler.parameters() if p.requires_grad]

        if params:
            lr = float(self.cfg.get("lr", 1e-4))
            opt = torch.optim.AdamW(params, lr=lr, weight_decay=float(self.cfg.get("weight_decay", 0.01)))
            sched = WarmupCosine(base_lr=lr, warmup_steps=int(self.cfg.get("warmup_steps", 1000)), total_steps=int(self.cfg.get("total_steps", 20000)), min_lr=float(self.cfg.get("min_lr", 1e-6)))
            amp = AMP(enabled=bool(self.cfg.get("amp", True)))
        else:
            opt = None
            sched = None
            amp = AMP(enabled=False)

        step = 0
        total_steps = int(self.cfg.get("total_steps", 20000))
        t0 = time.time()

        prev_x = None
        prev_erfg = None

        while step < total_steps:
            batch = next(it)
            video = batch["video"].to(self.device, non_blocking=True)

            x0 = _frame(video, 0)
            x1 = _frame(video, 1)

            with torch.no_grad():
                ev0 = encoder(x0, None)
                ev1 = encoder(x1, x0)

            # run compiler like demo
            w0 = compiler.step(ev0, dt=1.0 / 15.0)
            w1 = compiler.step(ev1, dt=1.0 / 15.0)

            erfg0 = w0["erfg"]
            erfg1 = w1["erfg"]

            p0 = _entity_tensor(erfg0).to(self.device)
            p1 = _entity_tensor(erfg1).to(self.device)

            loss = torch.tensor(0.0, device=self.device)

            if p0.numel() > 0 and p1.numel() > 0:
                n = min(p0.shape[0], p1.shape[0])
                loss_pos = F.smooth_l1_loss(p0[:n], p1[:n].detach())
                loss = loss + float(self.cfg.get("w_pos", 1.0)) * loss_pos

            # encourage sparse discrete events (not noisy)
            events1 = w1.get("events", [])
            if isinstance(events1, list):
                loss_evt = torch.tensor(float(len(events1)), device=self.device) * float(self.cfg.get("w_evt", 0.001))
                loss = loss + loss_evt

            if opt is not None:
                opt.zero_grad(set_to_none=True)
                amp.backward(loss)
                amp.unscale_(opt)
                gn = clip_grad(compiler, float(self.cfg.get("grad_clip", 1.0)))
                for pg in opt.param_groups:
                    pg["lr"] = sched.lr(step)
                amp.step(opt)
            else:
                gn = 0.0

            if step % int(self.cfg.get("log_every", 50)) == 0:
                self.logger.log({"stage": "stage2_compiler", "step": step, "loss": float(loss.item()), "grad_norm": float(gn), "sec": time.time() - t0})

            if step % int(self.cfg.get("ckpt_every", 2000)) == 0 and step > 0:
                payload = {"step": step}
                if hasattr(compiler, "state_dict"):
                    payload["compiler"] = compiler.state_dict()
                self.ckpt.save(f"stage2_compiler_step{step:07d}", payload)

            step += 1

        state["compiler"] = compiler
        return state
