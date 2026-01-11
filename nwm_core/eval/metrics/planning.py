from __future__ import annotations
from dataclasses import dataclass

@dataclass
class PlanningMetrics:
    success_rate: float
    avg_cost: float

def compute_planning_metrics(*args, **kwargs) -> PlanningMetrics:
    return PlanningMetrics(success_rate=0.0, avg_cost=0.0)
