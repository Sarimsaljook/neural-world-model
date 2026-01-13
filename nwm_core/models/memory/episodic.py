from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import time
import json
import torch


@dataclass(frozen=True)
class EpisodicConfig:
    max_episodes: int = 2000
    max_events_per_episode: int = 256
    store_raw: bool = False


class EpisodicMemory:
    def __init__(self, cfg: Optional[EpisodicConfig] = None):
        self.cfg = cfg or EpisodicConfig()
        self.episodes: List[Dict[str, Any]] = []

    def add(self, erfg: Any, events: List[Any], evidence: Optional[Dict[str, Any]] = None) -> None:
        snap = self._snapshot_erfg(erfg)
        ev = self._snapshot_events(events)

        ep = {
            "t_wall": float(time.time()),
            "t_world": float(getattr(erfg, "time", 0.0)),
            "erfg": snap,
            "events": ev[: self.cfg.max_events_per_episode],
        }
        if self.cfg.store_raw and evidence is not None:
            ep["evidence_keys"] = list(evidence.keys())
        self.episodes.append(ep)
        if len(self.episodes) > self.cfg.max_episodes:
            self.episodes = self.episodes[-self.cfg.max_episodes :]

    def last(self, k: int = 1) -> List[Dict[str, Any]]:
        return self.episodes[-k:]

    def dump_jsonl(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for ep in self.episodes:
                f.write(json.dumps(ep) + "\n")

    def _snapshot_erfg(self, erfg: Any) -> Dict[str, Any]:
        ents = getattr(erfg, "entities", {})
        rels = getattr(erfg, "relations", {})
        out_ents = []
        for eid, e in ents.items():
            pos = getattr(e, "pos", None) if not isinstance(e, dict) else e.get("pos", None)
            vel = getattr(e, "vel", None) if not isinstance(e, dict) else e.get("vel", None)
            cls = getattr(e, "cls", None) if not isinstance(e, dict) else e.get("cls", None)
            conf = getattr(e, "conf", None) if not isinstance(e, dict) else e.get("conf", None)
            out_ents.append(
                {
                    "id": int(eid),
                    "cls": int(cls) if cls is not None else None,
                    "conf": float(conf) if conf is not None else None,
                    "pos": torch.as_tensor(pos).float().tolist() if pos is not None else None,
                    "vel": torch.as_tensor(vel).float().tolist() if vel is not None else None,
                }
            )
        out = {"time": float(getattr(erfg, "time", 0.0)), "entities": out_ents, "relations": rels}
        return out

    def _snapshot_events(self, events: List[Any]) -> List[Dict[str, Any]]:
        out = []
        for e in events:
            if isinstance(e, dict):
                out.append(dict(e))
                continue
            out.append(
                {
                    "kind": getattr(e, "kind", ""),
                    "src": getattr(e, "src", None),
                    "dst": getattr(e, "dst", None),
                    "p": float(getattr(e, "p", 1.0)),
                    "params": getattr(e, "params", {}) if hasattr(e, "params") else {},
                }
            )
        return out
