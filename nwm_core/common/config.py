from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence

import yaml

def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out

def resolve_inherits(cfg: Dict[str, Any], cfg_dir: Path) -> Dict[str, Any]:
    inherits = cfg.get("inherits")
    if not inherits:
        return cfg
    if not isinstance(inherits, list):
        raise TypeError("inherits must be a list of file paths")
    base: Dict[str, Any] = {}
    for rel in inherits:
        p = (cfg_dir / rel).resolve()
        base_cfg = load_config(p)
        base = deep_merge(base, base_cfg)
    cfg2 = copy.deepcopy(cfg)
    cfg2.pop("inherits", None)
    return deep_merge(base, cfg2)

def _interpolate(obj: Any, ctx: Dict[str, Any]) -> Any:
    if isinstance(obj, dict):
        return {k: _interpolate(v, ctx) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate(v, ctx) for v in obj]
    if isinstance(obj, str) and "${" in obj:
        s = obj
        # minimal `${a.b.c}` resolver
        import re
        for m in re.findall(r"\$\{([^}]+)\}", s):
            path = m.strip()
            cur: Any = ctx
            for part in path.split("."):
                if not isinstance(cur, dict) or part not in cur:
                    cur = None
                    break
                cur = cur[part]
            if cur is None:
                continue
            s = s.replace("${" + m + "}", str(cur))
        return s
    return obj

def load_config(path: Path) -> Dict[str, Any]:
    cfg_dir = path.parent
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config must be a mapping: {path}")
    data = resolve_inherits(data, cfg_dir)
    # second pass interpolation with self-context
    data = _interpolate(data, data)
    return data
