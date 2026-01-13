from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .constraints import resolve_selector_to_id, ConstraintConfig


@dataclass(frozen=True)
class ProgramConfig:
    max_steps: int = 16
    min_conf: float = 0.55


@dataclass(frozen=True)
class EventStep:
    kind: str
    src: Optional[int]
    dst: Optional[int]
    params: Dict[str, float]
    p: float


@dataclass
class EventProgram:
    steps: List[EventStep]

    def __len__(self) -> int:
        return len(self.steps)


def _goal_from_language(mapped: Dict[str, Any], erfg: Any) -> Dict[str, Any]:
    g = dict(mapped.get("goal", {}) or {})
    intent = str(g.get("intent", "observe"))
    selectors = mapped.get("selectors", []) or []

    tgt = None
    surf = None

    if len(selectors) >= 1:
        tgt = resolve_selector_to_id(selectors[0], erfg, ConstraintConfig())
    if len(selectors) >= 2:
        surf = resolve_selector_to_id(selectors[1], erfg, ConstraintConfig())

    goal: Dict[str, Any] = {"kind": intent}
    if intent in {"grasp", "pick"}:
        if tgt is not None:
            goal["target_id"] = int(tgt)
    elif intent in {"place"}:
        if tgt is not None:
            goal["target_id"] = int(tgt)
        if surf is not None:
            goal["surface_id"] = int(surf)
    elif intent in {"pour"}:
        if tgt is not None:
            goal["source_id"] = int(tgt)
        if surf is not None:
            goal["target_id"] = int(surf)
    elif intent in {"push", "pull", "slide"}:
        if tgt is not None:
            goal["target_id"] = int(tgt)

    rb = None
    for c in mapped.get("constraints", []) or []:
        if str(c.get("type", "")) == "risk":
            rb = float(c.get("value", 0.25))
            break
    if rb is not None:
        goal["risk_budget"] = float(rb)

    return goal


def synthesize_program(
    erfg: Any,
    events: List[Any],
    goal: Optional[Dict[str, Any]] = None,
    cfg: Optional[ProgramConfig] = None,
) -> EventProgram:
    cfg = cfg or ProgramConfig()

    steps: List[EventStep] = []
    gk = None if goal is None else goal.get("kind", None)

    if gk in {"grasp", "pick"}:
        target = goal.get("target_id", None)
        steps.append(EventStep("find", None, target, {}, 1.0))
        steps.append(EventStep("approach", None, target, {}, 0.9))
        steps.append(EventStep("grasp_begin", None, target, {}, 0.9))
        steps.append(EventStep("lift", None, target, {"dz": 0.10}, 0.8))

    elif gk == "place":
        target = goal.get("target_id", None)
        surface = goal.get("surface_id", None)
        steps.append(EventStep("approach", None, target, {}, 0.9))
        steps.append(EventStep("move_over", target, surface, {}, 0.8))
        steps.append(EventStep("lower", None, target, {"dz": -0.10}, 0.8))
        steps.append(EventStep("release", None, target, {}, 0.85))

    elif gk == "pour":
        src = goal.get("source_id", None)
        dst = goal.get("target_id", None)
        steps.append(EventStep("grasp_begin", None, src, {}, 0.9))
        steps.append(EventStep("move_over", src, dst, {}, 0.8))
        steps.append(EventStep("tilt", src, dst, {"angle_deg": 35.0}, 0.75))
        steps.append(EventStep("until_contains", src, dst, {}, 0.7))
        steps.append(EventStep("tilt_back", src, dst, {"angle_deg": 0.0}, 0.7))

    elif gk in {"push", "pull", "slide"}:
        target = goal.get("target_id", None)
        steps.append(EventStep("approach", None, target, {}, 0.9))
        steps.append(EventStep("contact_begin", None, target, {}, 0.7))
        steps.append(EventStep("apply_force", None, target, {"fx": 0.3, "fy": 0.0, "fz": 0.0}, 0.65))
        steps.append(EventStep("contact_end", None, target, {}, 0.6))

    elif gk in {"openclose"}:
        target = goal.get("target_id", None)
        steps.append(EventStep("approach", None, target, {}, 0.9))
        steps.append(EventStep("grasp_begin", None, target, {}, 0.7))
        steps.append(EventStep("pull", None, target, {"dx": -0.1}, 0.6))

    elif gk in {"avoid", "observe"}:
        for e in events[: cfg.max_steps]:
            kind = e.get("kind", "") if isinstance(e, dict) else getattr(e, "kind", "")
            p = float(e.get("p", 1.0) if isinstance(e, dict) else getattr(e, "p", 1.0))
            if p < cfg.min_conf:
                continue
            src = e.get("src", None) if isinstance(e, dict) else getattr(e, "src", None)
            dst = e.get("dst", None) if isinstance(e, dict) else getattr(e, "dst", None)
            steps.append(EventStep(kind, src, dst, {}, p))
            if len(steps) >= cfg.max_steps:
                break

    return EventProgram(steps[: cfg.max_steps])


def synthesize_program_from_language(erfg: Any, events: List[Any], mapped: Dict[str, Any], cfg: Optional[ProgramConfig] = None) -> EventProgram:
    goal = _goal_from_language(mapped, erfg)
    return synthesize_program(erfg, events, goal=goal, cfg=cfg)


def program_to_text(prog: EventProgram) -> str:
    parts = []
    for s in prog.steps:
        a = "None" if s.src is None else str(s.src)
        b = "None" if s.dst is None else str(s.dst)
        parts.append(f"{s.kind} {a}->{b} p={s.p:.2f}")
    return "\n".join(parts)
