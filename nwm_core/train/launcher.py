from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from ..common.config import load_config
from ..common.logging import get_logger, setup_logging

log = get_logger("nwm.train")

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    args = p.parse_args()

    cfg = load_config(Path(args.config))
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    stage = cfg.get("stage", "")
    if not stage:
        raise ValueError("Config missing `stage`")

    if stage == "stage1_encoder":
        from .stages.stage1_encoder import run
    elif stage == "stage2_compiler":
        from .stages.stage2_compiler import run
    elif stage == "stage3_mechanisms":
        from .stages.stage3_mechanisms import run
    elif stage == "stage4_intuition":
        from .stages.stage4_intuition import run
    elif stage == "stage5_probe_planner":
        from .stages.stage5_probe_planner import run
    elif stage == "stage6_memory":
        from .stages.stage6_memory import run
    else:
        raise ValueError(f"Unknown stage: {stage}")

    run(cfg)

if __name__ == "__main__":
    main()
