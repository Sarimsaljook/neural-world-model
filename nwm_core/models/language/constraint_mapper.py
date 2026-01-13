from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .goal_parser import GoalSpec, EntityRef, RelationConstraint, NumericConstraint


@dataclass(frozen=True)
class ConstraintMapperConfig:
    default_p_min: float = 0.6
    max_constraints: int = 32
    allow_unbound_entities: bool = True


def _entity_selector(e: EntityRef) -> Dict:
    sel: Dict = {}
    if e.entity_id is not None:
        sel["entity_id"] = int(e.entity_id)
        return sel
    if e.class_name:
        sel["class"] = str(e.class_name)
    if e.text:
        sel["text"] = str(e.text)
    if e.attributes:
        sel["attrs"] = dict(e.attributes)
    return sel


def _map_relation(rc: RelationConstraint) -> Dict:
    return {
        "type": "relation",
        "kind": rc.kind,
        "src": _entity_selector(rc.src),
        "dst": _entity_selector(rc.dst),
        "p_min": float(rc.p_min),
        "params": dict(rc.params or {}),
    }


def _map_numeric(nc: NumericConstraint) -> Dict:
    return {
        "type": "numeric",
        "key": nc.key,
        "op": nc.op,
        "value": float(nc.value),
        "unit": nc.unit,
    }


def _risk_constraint(risk_budget: float) -> Dict:
    return {
        "type": "risk",
        "key": "risk_budget",
        "op": "<=",
        "value": float(risk_budget),
        "unit": None,
    }


class ConstraintMapper:
    def __init__(self, cfg: Optional[ConstraintMapperConfig] = None):
        self.cfg = cfg or ConstraintMapperConfig()

    def map(self, goal: GoalSpec) -> Dict:
        selectors = [_entity_selector(e) for e in goal.entities]

        constraints: List[Dict] = []
        avoid: List[Dict] = []

        constraints.append(_risk_constraint(goal.risk_budget))

        for r in goal.relations:
            rr = _map_relation(r)
            if rr["p_min"] is None:
                rr["p_min"] = self.cfg.default_p_min
            constraints.append(rr)

        for n in goal.numeric:
            constraints.append(_map_numeric(n))

        for r in goal.avoid:
            avoid.append(_map_relation(r))

        constraints = constraints[: self.cfg.max_constraints]
        avoid = avoid[: self.cfg.max_constraints]

        g = {
            "intent": goal.intent,
            "horizon_s": float(goal.horizon_s),
            "priority": int(goal.priority),
            "meta": dict(goal.meta or {}),
        }

        return {
            "raw_text": goal.raw_text,
            "goal": g,
            "selectors": selectors,
            "constraints": constraints,
            "avoid": avoid,
        }
