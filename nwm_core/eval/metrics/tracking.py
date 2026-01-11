from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

@dataclass
class TrackingMetrics:
    mota: float
    motp: float

def compute_tracking_metrics(*args, **kwargs) -> TrackingMetrics:
    return TrackingMetrics(mota=0.0, motp=0.0)
