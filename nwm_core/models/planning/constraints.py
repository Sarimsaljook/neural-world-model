from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass(frozen=True)
class ConstraintConfig:
    risk_max_default: float = 0.75
    collision_max_default: float = 0.65
    spill_max_default: float = 0.60
    require_known_support: bool = False
    default_weight: float = 1.0
    selector_match_min: float = 0.0


@dataclass(frozen=True)
class Constraint:
    kind: str
    weight: float
    params: Dict[str, float]
    selector: Optional[Dict[str, Any]] = None
    selector_b: Optional[Dict[str, Any]] = None


@dataclass
class ConstraintSet:
    items: List[Constraint]

    def get(self, kind: str) -> List[Constraint]:
        return [c for c in self.items if c.kind == kind]


def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _entity_fields(ent: Any) -> Dict[str, Any]:
    d: Dict[str, Any] = {}
    if isinstance(ent, dict):
        d = ent
    else:
        d = ent.__dict__ if hasattr(ent, "__dict__") else {}

    out: Dict[str, Any] = {}
    out["id"] = d.get("id", d.get("entity_id", None))
    out["class"] = d.get("class_name", d.get("cls", d.get("type", None)))
    out["text"] = d.get("name", d.get("text", None))
    out["attrs"] = d.get("attributes", d.get("attrs", {})) or {}
    return out


def _match_score(selector: Dict[str, Any], ent: Any) -> float:
    if selector is None:
        return 0.0

    ef = _entity_fields(ent)
    sid = selector.get("entity_id", None)
    if sid is not None and ef.get("id", None) is not None:
        return 1.0 if int(sid) == int(ef["id"]) else -1.0

    score = 0.0
    scls = selector.get("class", None)
    if scls:
        if _norm(str(scls)) == _norm(str(ef.get("class", ""))):
            score += 1.0
        else:
            score += 0.0

    stxt = selector.get("text", None)
    if stxt:
        if _norm(str(stxt)) in _norm(str(ef.get("text", ""))) or _norm(str(ef.get("text", ""))) in _norm(str(stxt)):
            score += 0.5

    sattrs = selector.get("attrs", None) or {}
    if isinstance(sattrs, dict) and sattrs:
        eattrs = ef.get("attrs", {}) or {}
        hit = 0
        tot = 0
        for k, v in sattrs.items():
            tot += 1
            if k in eattrs and _norm(str(eattrs[k])) == _norm(str(v)):
                hit += 1
        if tot > 0:
            score += 0.5 * (hit / float(tot))

    return float(score)


def resolve_selector_to_id(selector: Dict[str, Any], erfg: Any, cfg: Optional[ConstraintConfig] = None) -> Optional[int]:
    cfg = cfg or ConstraintConfig()
    ents = getattr(erfg, "entities", None)

    if ents is None:
        return None

    if isinstance(ents, dict):
        ent_list = list(ents.values())
    elif isinstance(ents, list):
        ent_list = ents
    else:
        try:
            ent_list = list(ents)
        except Exception:
            return None

    best_id = None
    best_score = -1e9
    for e in ent_list:
        sc = _match_score(selector, e)
        if sc > best_score:
            best_score = sc
            ef = _entity_fields(e)
            best_id = ef.get("id", None)

    if best_id is None:
        return None
    if best_score < cfg.selector_match_min:
        return None
    return int(best_id)


def build_constraint_set(
    parsed_goal: Optional[Dict[str, Any]] = None,
    mapped_constraints: Optional[List[Dict[str, Any]]] = None,
    mapped_avoid: Optional[List[Dict[str, Any]]] = None,
    cfg: Optional[ConstraintConfig] = None,
) -> ConstraintSet:
    cfg = cfg or ConstraintConfig()
    out: List[Constraint] = []

    out.append(Constraint("risk_max", cfg.default_weight, {"max": float(cfg.risk_max_default)}))
    out.append(Constraint("collision_max", cfg.default_weight, {"max": float(cfg.collision_max_default)}))
    out.append(Constraint("spill_max", cfg.default_weight, {"max": float(cfg.spill_max_default)}))
    if cfg.require_known_support:
        out.append(Constraint("require_known_support", cfg.default_weight, {"min_p": 0.6}))

    if parsed_goal and "risk_budget" in parsed_goal:
        rb = float(parsed_goal["risk_budget"])
        out.append(Constraint("risk_budget", 1.0, {"max": rb}))

    def add_mapped(items: Optional[List[Dict[str, Any]]], sign: float) -> None:
        if not items:
            return
        for c in items:
            t = str(c.get("type", ""))
            if t == "risk":
                mx = float(c.get("value", cfg.risk_max_default))
                out.append(Constraint("risk_budget", 1.0 * sign, {"max": mx}))
                continue
            if t == "numeric":
                key = str(c.get("key", ""))
                op = str(c.get("op", "<="))
                val = float(c.get("value", 0.0))
                unit = c.get("unit", None)
                kind = f"numeric:{key}:{op}:{unit or ''}"
                out.append(Constraint(kind, 1.0 * sign, {"value": val}))
                continue
            if t == "relation":
                kind = str(c.get("kind", ""))
                pmin = float(c.get("p_min", 0.6))
                src = c.get("src", None)
                dst = c.get("dst", None)
                params = dict(c.get("params", {}) or {})
                params["p_min"] = pmin
                out.append(Constraint(f"rel:{kind}", 1.0 * sign, params, selector=src, selector_b=dst))

    add_mapped(mapped_constraints, +1.0)
    add_mapped(mapped_avoid, -1.0)

    return ConstraintSet(out)


def score_constraints(constraints: ConstraintSet, intuition: Optional[Dict[str, Any]], erfg: Any) -> float:
    viol = constraint_violations(constraints, intuition, erfg)
    s = 0.0
    for _, v in viol:
        s += float(v)
    return float(s)


def constraint_violations(constraints: ConstraintSet, intuition: Optional[Dict[str, Any]], erfg: Any) -> List[Tuple[str, float]]:
    intuition = intuition or {}
    out: List[Tuple[str, float]] = []

    risk = intuition.get("risk", None) or intuition.get("risk_map", None) or {}
    stability = intuition.get("stability", {}) or {}
    slip = intuition.get("slip", {}) or {}
    collisions = intuition.get("collision_pairs", []) or []

    max_risk = 0.0
    if isinstance(risk, dict):
        for _, s in risk.items():
            max_risk = max(max_risk, float(s))

    max_slip = 0.0
    if isinstance(slip, dict):
        for _, s in slip.items():
            max_slip = max(max_slip, float(s))

    max_unstable = 0.0
    if isinstance(stability, dict):
        for _, s in stability.items():
            max_unstable = max(max_unstable, float(s))

    col_score = 0.0
    if isinstance(collisions, list) and len(collisions) > 0:
        for p in collisions:
            if isinstance(p, dict):
                col_score = max(col_score, float(p.get("p", 0.0)))
            elif isinstance(p, (list, tuple)) and len(p) >= 3:
                col_score = max(col_score, float(p[2]))

    for c in constraints.items:
        if c.kind == "risk_max":
            mx = float(c.params.get("max", 1.0))
            out.append(("risk_max", max(0.0, max_risk - mx) * abs(float(c.weight))))
        elif c.kind == "risk_budget":
            mx = float(c.params.get("max", 1.0))
            out.append(("risk_budget", max(0.0, max_risk - mx) * abs(float(c.weight))))
        elif c.kind == "collision_max":
            mx = float(c.params.get("max", 1.0))
            out.append(("collision_max", max(0.0, col_score - mx) * abs(float(c.weight))))
        elif c.kind == "spill_max":
            mx = float(c.params.get("max", 1.0))
            out.append(("spill_max", max(0.0, max_slip - mx) * abs(float(c.weight))))
        elif c.kind == "require_known_support":
            minp = float(c.params.get("min_p", 0.6))
            rels = getattr(erfg, "relations", None)
            ok = False
            if isinstance(rels, list):
                for r in rels:
                    k = r.get("kind", "") if isinstance(r, dict) else getattr(r, "kind", "")
                    p = float(r.get("p", 0.0) if isinstance(r, dict) else getattr(r, "p", 0.0))
                    if k in {"supporting", "on", "supports"} and p >= minp:
                        ok = True
                        break
            out.append(("require_known_support", 0.0 if ok else 1.0 * abs(float(c.weight))))
        elif c.kind.startswith("rel:"):
            out.append((c.kind, 0.0))
        elif c.kind.startswith("numeric:"):
            out.append((c.kind, 0.0))

    return out
