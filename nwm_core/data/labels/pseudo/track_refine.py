from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class TrackRefiner:
    min_track_len: int = 5

    def refine(self, tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [t for t in tracks if int(t.get("length", 0)) >= self.min_track_len]
