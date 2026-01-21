from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nwm_core.train.checkpoints.model_zoo import build_core
from nwm_core.train.optim import AMP, WarmupCosine, clip_grad
from nwm_core.train.stages._data import build_datapipe

class TrainableAdapter(nn.Module):
    def __init__(self, input_dim=768, output_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderWithAdapter(nn.Module):
    def __init__(self, frozen_encoder: nn.Module, adapter: nn.Module):
        super().__init__()
        self.frozen_encoder = frozen_encoder
        self.adapter = adapter

    def train(self, mode=True):
        self.frozen_encoder.eval()
        self.adapter.train(mode)
        return self

    def forward(self, x: torch.Tensor, prev: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        with torch.no_grad():
            out = self.frozen_encoder(x, prev)

        inst = out.get("instances", {})
        if "embeddings" in inst and isinstance(inst["embeddings"], torch.Tensor):
            raw_emb = inst["embeddings"]
            if raw_emb.numel() > 0:
                inst["embeddings"] = self.adapter(raw_emb)
        return out


# ------------------------------------------------------------


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
        print(f"[{self.__class__.__name__}] Initializing...")

        # --- 1. Load Encoder (From State OR Checkpoint) ---
        if "encoder" in state:
            print("Using encoder passed from previous stage.")
            encoder = state["encoder"].to(self.device).eval()
            compiler = state["compiler"]
        else:
            print("Encoder not found in state. Loading from checkpoint...")

            # Build Base Core
            core = build_core(self.device, cfg=self.cfg.get("model", {}))
            frozen_encoder = core["encoder"].eval()
            compiler = core["compiler"]

            # Reconstruct the Adapter structure
            adapter = TrainableAdapter().to(self.device)
            encoder = EncoderWithAdapter(frozen_encoder, adapter).to(self.device)

            # Load Weights
            # UPDATE THIS PATH if it changes
            ckpt_path = "assets/runs/ssv2_nwm/checkpoints/stage1_adapter_step0008000.pt"

            # Allow config to override this hardcoded path
            if self.cfg.get("init", {}).get("encoder_ckpt"):
                ckpt_path = self.cfg.get("init", {}).get("encoder_ckpt")

            print(f"Loading weights from: {ckpt_path}")
            if Path(ckpt_path).exists():
                payload = torch.load(ckpt_path, map_location=self.device)
                if "adapter" in payload:
                    adapter.load_state_dict(payload["adapter"])
                    print("Adapter weights loaded successfully.")
                elif "encoder" in payload:
                    # Fallback if you saved the whole encoder object
                    encoder.load_state_dict(payload["encoder"])
            else:
                raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

            encoder.eval()
        # --------------------------------------------------

        pipe = build_datapipe(self.data_cfg, self.cfg)
        it = pipe.train_iter()

        # compiler may or may not have parameters; support both
        params = []
        if hasattr(compiler, "parameters"):
            params = [p for p in compiler.parameters() if p.requires_grad]

        if params:
            lr = float(self.cfg.get("lr", 1e-4))
            opt = torch.optim.AdamW(params, lr=lr, weight_decay=float(self.cfg.get("weight_decay", 0.01)))
            sched = WarmupCosine(base_lr=lr, warmup_steps=int(self.cfg.get("warmup_steps", 1000)),
                                 total_steps=int(self.cfg.get("total_steps", 20000)),
                                 min_lr=float(self.cfg.get("min_lr", 1e-6)))
            amp = AMP(enabled=bool(self.cfg.get("amp", True)))
        else:
            opt = None
            sched = None
            amp = AMP(enabled=False)

        step = 0
        total_steps = int(self.cfg.get("total_steps", 20000))
        t0 = time.time()

        while step < total_steps:
            try:
                batch = next(it)
            except StopIteration:
                it = pipe.train_iter()
                batch = next(it)

            video = batch["video"].to(self.device, non_blocking=True)

            x0 = _frame(video, 0)
            x1 = _frame(video, 1)

            # Encoder is now the "Eyes" - Frozen
            with torch.no_grad():
                ev0 = encoder(x0, None)
                ev1 = encoder(x1, x0)

            # Compiler is the "Brain" - Trainable
            # step() usually takes evidence and updates internal world graph
            w0 = compiler.step(ev0, dt=1.0 / 15.0)
            w1 = compiler.step(ev1, dt=1.0 / 15.0)

            erfg0 = w0["erfg"]
            erfg1 = w1["erfg"]

            p0 = _entity_tensor(erfg0).to(self.device)
            p1 = _entity_tensor(erfg1).to(self.device)

            loss = torch.tensor(0.0, device=self.device)
            valid_loss = False

            # 1. Position Consistency Loss
            if p0.numel() > 0 and p1.numel() > 0:
                n = min(p0.shape[0], p1.shape[0])
                loss_pos = F.smooth_l1_loss(p0[:n], p1[:n].detach())
                loss = loss + float(self.cfg.get("w_pos", 1.0)) * loss_pos
                valid_loss = True

            # 2. Event Sparsity Loss (encourage sparse discrete events)
            events1 = w1.get("events", [])
            if isinstance(events1, list) and len(events1) > 0:
                loss_evt = torch.tensor(float(len(events1)), device=self.device) * float(self.cfg.get("w_evt", 0.001))
                loss = loss + loss_evt
                # Note: this is a weak signal, usually we want positive event supervision too
                valid_loss = True

            if not valid_loss:
                # If compiler produced no entities/events, skip update to avoid crash
                if step % 50 == 0:
                    print(f"Step {step}: Warning - No entities/events tracked. Skipping.")
                step += 1
                opt.zero_grad(set_to_none=True)
                continue

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
                dt = time.time() - t0
                self.logger.log({
                    "stage": "stage2_compiler",
                    "step": step,
                    "loss": float(loss.item()),
                    "grad_norm": float(gn),
                    "lr": sched.lr(step) if sched else 0.0,
                    "sec": dt
                })
                t0 = time.time()  # Reset timer

            if step % int(self.cfg.get("ckpt_every", 2000)) == 0 and step > 0:
                payload = {"step": step}
                if hasattr(compiler, "state_dict"):
                    payload["compiler"] = compiler.state_dict()
                self.ckpt.save(f"stage2_compiler_step{step:07d}", payload)

            step += 1

        state["encoder"] = encoder
        state["compiler"] = compiler
        return state