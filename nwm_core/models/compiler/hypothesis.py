from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TrackHypothesis:
    track_id: int
    weight: float
    missed: int


class HypothesisBank:
    def __init__(self, max_hypotheses: int = 6):
        self.max_hypotheses = int(max_hypotheses)
        self.hypotheses: List[TrackHypothesis] = []

    def decay(self, decay: float = 0.98) -> None:
        for h in self.hypotheses:
            h.weight *= float(decay)

    def add_or_update(self, track_id: int, weight: float, missed: int) -> None:
        for h in self.hypotheses:
            if h.track_id == track_id:
                h.weight = float(weight)
                h.missed = int(missed)
                return
        self.hypotheses.append(TrackHypothesis(track_id=track_id, weight=float(weight), missed=int(missed)))
        self.hypotheses.sort(key=lambda x: -x.weight)
        self.hypotheses = self.hypotheses[: self.max_hypotheses]

    def best(self) -> Optional[int]:
        if not self.hypotheses:
            return None
        return self.hypotheses[0].track_id
