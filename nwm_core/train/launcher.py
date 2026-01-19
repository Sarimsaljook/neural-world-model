from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

from nwm_core.train.checkpoints.io import Checkpointer
from nwm_core.train.reports.curves import CurveLogger
from nwm_core.train.stages.stage1_encoder import Stage1Encoder
from nwm_core.train.stages.stage2_compiler import Stage2Compiler
from nwm_core.train.stages.stage3_mechanisms import Stage3Mechanisms
from nwm_core.train.stages.stage4_intuition import Stage4Intuition
from nwm_core.train.stages.stage5_probe_planner import Stage5ProbePlanner
from nwm_core.train.stages.stage6_memory import Stage6Memory


def _set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _load_yaml(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


_STAGE_REGISTRY = {
    "stage1_encoder": Stage1Encoder,
    "stage2_compiler": Stage2Compiler,
    "stage3_mechanisms": Stage3Mechanisms,
    "stage4_intuition": Stage4Intuition,
    "stage5_probe_planner": Stage5ProbePlanner,
    "stage6_memory": Stage6Memory,
}


def _resolve_stage_name(cfg: Dict[str, Any]) -> str:
    name = str(cfg.get("stage", {}).get("name", "")).strip()
    if name:
        return name
    raise ValueError("Stage config must include a 'name' field (e.g. stage1_encoder).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--stages", type=str, nargs="+", required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--workdir", type=str, default="assets/runs/nwm_train")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    _set_seed(args.seed)

    data_cfg = _load_yaml(args.data)
    stage_cfgs = [_load_yaml(p) for p in args.stages]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    logger = CurveLogger(workdir / "metrics.jsonl")

    ckpt = Checkpointer(workdir / "checkpoints")
    ckpt.root.mkdir(parents=True, exist_ok=True)

    state: Dict[str, Any] = {"device": device, "seed": args.seed}

    for scfg in stage_cfgs:
        stage_name = _resolve_stage_name(scfg)
        if stage_name not in _STAGE_REGISTRY:
            raise ValueError(f"Unknown stage: {stage_name}. Options: {list(_STAGE_REGISTRY.keys())}")

        StageCls = _STAGE_REGISTRY[stage_name]
        stage = StageCls(device=device, data_cfg=data_cfg, stage_cfg=scfg, workdir=workdir, logger=logger, ckpt=ckpt)

        state = stage.run(state)

    final_path = outdir / "nwm.pt"
    ckpt.export_consolidated(final_path, state)
    print(f"[train] export -> {final_path}")


if __name__ == "__main__":
    main()
