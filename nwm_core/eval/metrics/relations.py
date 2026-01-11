from __future__ import annotations
from dataclasses import dataclass

@dataclass
class RelationMetrics:
    f1: float
    precision: float
    recall: float

def compute_relation_metrics(*args, **kwargs) -> RelationMetrics:
    return RelationMetrics(f1=0.0, precision=0.0, recall=0.0)
