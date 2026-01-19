from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F

from nwm_core.train.checkpoints.model_zoo import build_core
from nwm_core.train.optim import AMP, WarmupCosine, clip_grad
from nwm_core.train.stages._data import build_datapipe


def _frame_batch(video: torch.Tensor, t: int) -> torch.Tensor:
    # video: (B,T,3,H,W) -> (B,3,H,W)
    return video[:, t]


def _safe_get(d: Dict[str, Any], k: str, default=None):
    return d.get(k, default) if isinstance(d, dict) else default


class Stage1Encoder:
    def __init__(self, device, data_cfg, stage_cfg, workdir: Path, logger, ckpt):
        self.device = device
        self.data_cfg = data_cfg
        self.cfg = stage_cfg
        self.workdir = workdir
        self.logger = logger
        self.ckpt = ckpt

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        core = build_core(self.device, cfg=self.cfg.get("model", {}))
        encoder = core["encoder"].train()

        pipe = build_datapipe(self.data_cfg, self.cfg)
        it = pipe.train_iter()

        lr = float(self.cfg.get("lr", 3e-4))
        wd = float(self.cfg.get("weight_decay", 0.05))
        opt = torch.optim.AdamW([p for p in encoder.parameters() if p.requires_grad], lr=lr, weight_decay=wd)

        total_steps = int(self.cfg.get("total_steps", 50000))
        sched = WarmupCosine(base_lr=lr, warmup_steps=int(self.cfg.get("warmup_steps", 2000)), total_steps=total_steps, min_lr=float(self.cfg.get("min_lr", 3e-6)))

        amp = AMP(enabled=bool(self.cfg.get("amp", True)))
        grad_max = float(self.cfg.get("grad_clip", 1.0))

        step = 0
        t0 = time.time()
        while step < total_steps:
            batch = next(it)
            video = batch["video"].to(self.device, non_blocking=True)  # (B,T,3,H,W)

            # two consecutive frames
            x0 = _frame_batch(video, 0)
            x1 = _frame_batch(video, 1)

            with amp.autocast():
                out0 = encoder(x0, None)
                out1 = encoder(x1, x0)

                # Use instance embeddings if present, else fallback to global features if present
                inst0 = _safe_get(out0, "instances", {})
                inst1 = _safe_get(out1, "instances", {})

                e0 = _safe_get(inst0, "embeddings", None)
                e1 = _safe_get(inst1, "embeddings", None)

                loss = torch.tensor(0.0, device=self.device)

                if isinstance(e0, torch.Tensor) and isinstance(e1, torch.Tensor) and e0.numel() > 0 and e1.numel() > 0:
                    # (B,Q,D)
                    q = min(e0.shape[1], e1.shape[1], int(self.cfg.get("topk", 16)))
                    a = F.normalize(e0[:, :q], dim=-1)
                    b = F.normalize(e1[:, :q], dim=-1)
                    loss_emb = (1.0 - (a * b).sum(dim=-1)).mean()
                    loss = loss + float(self.cfg.get("w_emb", 1.0)) * loss_emb

                # feature predictive head if provided by encoder output
                f0 = _safe_get(out0, "feat", None)
                f1 = _safe_get(out1, "feat", None)
                if isinstance(f0, torch.Tensor) and isinstance(f1, torch.Tensor) and f0.numel() > 0 and f1.numel() > 0:
                    loss_feat = F.smooth_l1_loss(f0, f1.detach())
                    loss = loss + float(self.cfg.get("w_feat", 0.5)) * loss_feat

            opt.zero_grad(set_to_none=True)
            amp.backward(loss)
            amp.unscale_(opt)
            gn = clip_grad(encoder, grad_max)
            for pg in opt.param_groups:
                pg["lr"] = sched.lr(step)
            amp.step(opt)

            if step % int(self.cfg.get("log_every", 50)) == 0:
                dt = time.time() - t0
                self.logger.log({"stage": "stage1_encoder", "step": step, "loss": float(loss.item()), "grad_norm": float(gn), "lr": sched.lr(step), "sec": dt})

            if step % int(self.cfg.get("ckpt_every", 2000)) == 0 and step > 0:
                self.ckpt.save(f"stage1_encoder_step{step:07d}", {"encoder": encoder.state_dict(), "step": step})

            step += 1

        encoder.eval()
        state["encoder"] = encoder
        state["compiler"] = core["compiler"]
        return state
