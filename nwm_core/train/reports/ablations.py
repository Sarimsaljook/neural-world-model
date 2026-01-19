from __future__ import annotations

from typing import Dict, Any


def ablation_flags(stage_cfg: Dict[str, Any]) -> Dict[str, bool]:
    return dict(stage_cfg.get("ablate", {}) or {})
