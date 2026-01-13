from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class GoalParserConfig:
    max_entities: int = 6
    max_relations: int = 12
    default_horizon_s: float = 3.0
    default_risk_budget: float = 0.25
    default_priority: int = 5


@dataclass(frozen=True)
class EntityRef:
    text: str
    class_name: Optional[str] = None
    entity_id: Optional[int] = None
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class NumericConstraint:
    key: str
    op: str
    value: float
    unit: Optional[str] = None


@dataclass(frozen=True)
class RelationConstraint:
    kind: str
    src: EntityRef
    dst: EntityRef
    p_min: float = 0.5
    params: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class GoalSpec:
    raw_text: str
    intent: str
    entities: List[EntityRef]
    relations: List[RelationConstraint]
    numeric: List[NumericConstraint]
    avoid: List[RelationConstraint]
    horizon_s: float
    risk_budget: float
    priority: int
    meta: Dict[str, str] = field(default_factory=dict)


_NUM_PAT = re.compile(
    r"(?P<op>(<=|>=|<|>|=))?\s*(?P<val>\d+(\.\d+)?)\s*(?P<unit>cm|mm|m|meters?|seconds?|s|ms|degrees?|deg|%)?\b",
    flags=re.IGNORECASE,
)

_TIME_PAT = re.compile(r"\b(in|within|for)\s+(?P<val>\d+(\.\d+)?)\s*(?P<unit>seconds?|s|ms|minutes?|min)\b", flags=re.IGNORECASE)

_RISK_PAT = re.compile(r"\b(risk)\s*(<=|<|=)?\s*(?P<val>\d+(\.\d+)?)\s*(?P<unit>%|percent)?\b", flags=re.IGNORECASE)

_NEG_WORDS = {"don't", "do not", "avoid", "never", "no"}
_STOP = {
    "the", "a", "an", "to", "of", "on", "in", "at", "by", "with", "and", "or", "then", "please", "now",
    "make", "keep", "ensure", "try", "want", "need", "should", "must", "can", "could", "would",
}


def _norm(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _tokenize(s: str) -> List[str]:
    s = _norm(s)
    s = re.sub(r"[^a-z0-9%<>=\.\-\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return [t for t in s.split(" ") if t]


def _extract_horizon(text: str, default_s: float) -> float:
    m = _TIME_PAT.search(text.lower())
    if not m:
        return float(default_s)
    v = float(m.group("val"))
    u = (m.group("unit") or "s").lower()
    if u.startswith("ms"):
        return v / 1000.0
    if u.startswith("min"):
        return v * 60.0
    return v


def _extract_risk(text: str, default_budget: float) -> float:
    m = _RISK_PAT.search(text.lower())
    if not m:
        return float(default_budget)
    v = float(m.group("val"))
    u = (m.group("unit") or "").lower()
    if u in {"%", "percent"}:
        v = v / 100.0
    return float(max(0.0, min(1.0, v)))


def _extract_numeric(text: str) -> List[NumericConstraint]:
    out: List[NumericConstraint] = []
    t = text.lower()

    if "distance" in t or "far" in t or "near" in t:
        for m in _NUM_PAT.finditer(t):
            op = m.group("op") or "<="
            val = float(m.group("val"))
            unit = (m.group("unit") or "m").lower()
            if unit in {"mm"}:
                val = val / 1000.0
                unit = "m"
            elif unit in {"cm"}:
                val = val / 100.0
                unit = "m"
            elif unit in {"meters", "meter"}:
                unit = "m"
            out.append(NumericConstraint(key="distance_m", op=op, value=float(val), unit=unit))
            break

    if "speed" in t or "slow" in t or "fast" in t:
        m = re.search(r"\b(speed)\s*(<=|<|=|>=|>)?\s*(\d+(\.\d+)?)\s*(m/s|ms)\b", t)
        if m:
            op = m.group(2) or "<="
            val = float(m.group(3))
            unit = m.group(5).lower()
            if unit == "ms":
                unit = "m/s"
            out.append(NumericConstraint(key="speed_mps", op=op, value=val, unit=unit))

    if "angle" in t or "rotate" in t or "turn" in t:
        m = re.search(r"\b(angle)\s*(<=|<|=|>=|>)?\s*(\d+(\.\d+)?)\s*(deg|degree|degrees)\b", t)
        if m:
            op = m.group(2) or "<="
            val = float(m.group(3))
            out.append(NumericConstraint(key="angle_deg", op=op, value=val, unit="deg"))

    return out


def _contains_negation(tokens: List[str]) -> bool:
    joined = " ".join(tokens)
    return any(w in joined for w in _NEG_WORDS)


def _guess_intent(tokens: List[str]) -> str:
    t = set(tokens)
    if "grasp" in t or "pick" in t or "pickup" in t or "hold" in t:
        return "grasp"
    if "place" in t or "put" in t or "set" in t or "drop" in t:
        return "place"
    if "push" in t or "pull" in t or "slide" in t:
        return "push"
    if "open" in t or "close" in t:
        return "openclose"
    if "pour" in t or "spill" in t or "fill" in t:
        return "pour"
    if "avoid" in t or "don't" in t or "never" in t:
        return "avoid"
    return "observe"


def _extract_entity_phrases(text: str, max_entities: int) -> List[str]:
    t = _norm(text)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    for w in sorted(_STOP, key=len, reverse=True):
        t = re.sub(rf"\b{re.escape(w)}\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return []

    parts = re.split(r"\b(then|while|after|before|and|or)\b", t)
    phrases = []
    for p in parts:
        p = p.strip()
        if not p or p in {"then", "while", "after", "before", "and", "or"}:
            continue
        phrases.append(p)
        if len(phrases) >= max_entities:
            break
    return phrases


def _entity_from_phrase(phrase: str) -> EntityRef:
    phrase = _norm(phrase)
    phrase = re.sub(r"\s+", " ", phrase).strip()
    if not phrase:
        return EntityRef(text="object")

    attrs: Dict[str, str] = {}
    colors = ["red", "blue", "green", "black", "white", "yellow", "pink", "purple", "orange", "gray", "grey", "brown"]
    for c in colors:
        if re.search(rf"\b{c}\b", phrase):
            attrs["color"] = c
            phrase = re.sub(rf"\b{c}\b", "", phrase).strip()
            break

    sizes = ["small", "tiny", "big", "large", "huge"]
    for s in sizes:
        if re.search(rf"\b{s}\b", phrase):
            attrs["size"] = s
            phrase = re.sub(rf"\b{s}\b", "", phrase).strip()
            break

    phrase = re.sub(r"\s+", " ", phrase).strip()
    class_name = phrase if phrase else None
    return EntityRef(text=phrase or "object", class_name=class_name, entity_id=None, attributes=attrs)


def _relation_from_text(text: str, entities: List[EntityRef], max_rel: int) -> Tuple[List[RelationConstraint], List[RelationConstraint]]:
    t = _norm(text)
    rels: List[RelationConstraint] = []
    avoid: List[RelationConstraint] = []

    if len(entities) == 0:
        return rels, avoid

    a = entities[0]
    b = entities[1] if len(entities) > 1 else EntityRef(text="surface", class_name="table")

    neg = _contains_negation(_tokenize(text))

    def add(kind: str, src: EntityRef, dst: EntityRef, p: float):
        rc = RelationConstraint(kind=kind, src=src, dst=dst, p_min=p, params={})
        (avoid if neg else rels).append(rc)

    if any(w in t for w in ["on top of", "on", "onto"]):
        add("on", a, b, 0.6)
    if any(w in t for w in ["in", "inside", "into"]):
        add("inside", a, b, 0.6)
    if any(w in t for w in ["near", "close to"]):
        add("near", a, b, 0.55)
    if any(w in t for w in ["away", "far from"]):
        rc = RelationConstraint(kind="far", src=a, dst=b, p_min=0.55, params={})
        (avoid if not neg else rels).append(rc)
    if any(w in t for w in ["touch", "contact"]):
        add("contact", a, b, 0.6)
    if any(w in t for w in ["don't drop", "do not drop", "avoid drop"]):
        avoid.append(RelationConstraint(kind="drop", src=a, dst=EntityRef(text="ground", class_name="floor"), p_min=0.6, params={}))

    rels = rels[:max_rel]
    avoid = avoid[:max_rel]
    return rels, avoid


class GoalParser:
    def __init__(self, cfg: Optional[GoalParserConfig] = None):
        self.cfg = cfg or GoalParserConfig()

    def parse(self, text: str) -> GoalSpec:
        raw = text or ""
        tokens = _tokenize(raw)
        intent = _guess_intent(tokens)

        horizon_s = _extract_horizon(raw, self.cfg.default_horizon_s)
        risk_budget = _extract_risk(raw, self.cfg.default_risk_budget)

        numeric = _extract_numeric(raw)

        phrases = _extract_entity_phrases(raw, self.cfg.max_entities)
        entities = [_entity_from_phrase(p) for p in phrases]
        if len(entities) == 0:
            entities = [EntityRef(text="object", class_name=None)]

        relations, avoid = _relation_from_text(raw, entities, self.cfg.max_relations)

        priority = self.cfg.default_priority
        if "urgent" in tokens or "now" in tokens:
            priority = 9
        elif "later" in tokens:
            priority = 3

        meta: Dict[str, str] = {}
        if "careful" in tokens or "gently" in tokens:
            meta["style"] = "careful"
        if "fast" in tokens:
            meta["style"] = "fast"

        return GoalSpec(
            raw_text=raw,
            intent=intent,
            entities=entities,
            relations=relations,
            numeric=numeric,
            avoid=avoid,
            horizon_s=horizon_s,
            risk_budget=risk_budget,
            priority=int(priority),
            meta=meta,
        )
