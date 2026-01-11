from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from ..erfg.io import erfg_to_json
from ..erfg.state import ERFGState


@dataclass
class EpisodicMemory:
    root: Path
    max_per_day: int = 5000

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def _day_path(self, day: str) -> Path:
        return self.root / f"episodes_{day}.jsonl"

    def append(self, day: str, state: ERFGState, extras: Optional[Dict[str, Any]] = None) -> None:
        rec = {"state": erfg_to_json(state), "extras": extras or {}}
        p = self._day_path(day)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def iter_day(self, day: str) -> Iterator[Dict[str, Any]]:
        p = self._day_path(day)
        if not p.exists():
            return
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
