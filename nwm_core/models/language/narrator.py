from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _safe_get(x: Any, k: str, default=None):
    if isinstance(x, dict):
        return x.get(k, default)
    return getattr(x, k, default)


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def _entity_iter(erfg: Any) -> List[Any]:
    ents = _safe_get(erfg, "entities", None)
    if ents is None:
        return []
    if isinstance(ents, dict):
        return list(ents.values())
    if isinstance(ents, list):
        return ents
    try:
        return list(ents)
    except Exception:
        return []


def _max_dict(d: Any) -> float:
    if not isinstance(d, dict):
        return 0.0
    m = 0.0
    for _, v in d.items():
        try:
            m = max(m, float(v))
        except Exception:
            pass
    return float(m)


def _topk_dict(d: Dict[Any, Any], k: int = 3) -> List[Tuple[Any, float]]:
    items: List[Tuple[Any, float]] = []
    for kk, vv in (d or {}).items():
        try:
            items.append((kk, float(vv)))
        except Exception:
            pass
    items.sort(key=lambda t: t[1], reverse=True)
    return items[:k]


def _bucket(v: float) -> int:
    if v < 0.15:
        return 0
    if v < 0.35:
        return 1
    if v < 0.55:
        return 2
    if v < 0.75:
        return 3
    return 4


def _human_level(v: float) -> str:
    if v < 0.15:
        return "low"
    if v < 0.35:
        return "mild"
    if v < 0.55:
        return "moderate"
    if v < 0.75:
        return "high"
    return "very high"


def _shorten(s: str, n: int) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    return s[: n - 3].rstrip() + "..."


@dataclass(frozen=True)
class NarratorConfig:
    min_interval_s: float = 1.1
    max_sentences: int = 3

    # Speak triggers
    say_on_events: bool = True
    say_on_goal: bool = True
    say_on_risk: bool = True
    say_on_reco: bool = True

    risk_thresh: float = 0.35
    unstable_thresh: float = 0.55
    slip_thresh: float = 0.35

    cscore_delta_say: float = 0.12

    # Style
    natural: bool = True


class NWMNarrator:
    """
    Natural narrator that stays grounded:
    - entity naming (class labels instead of raw ids)
    - event meaning
    - belief summary
    - risk + recommendation
    - goal progress commentary
    """

    def __init__(self, cfg: Optional[NarratorConfig] = None):
        self.cfg = cfg or NarratorConfig()

        self._t_last = 0.0
        self._last_event_key: Optional[str] = None
        self._last_risk_bucket: int = -1
        self._last_unstable_bucket: int = -1
        self._last_slip_bucket: int = -1
        self._last_cscore: Optional[float] = None

        self._last_scene_hash: Optional[str] = None
        self._scene_cooldown: float = 0.0

    # ---------------------------
    # Naming + grounding helpers
    # ---------------------------

    def _entity_id(self, e: Any) -> Optional[int]:
        v = _safe_get(e, "id", _safe_get(e, "entity_id", None))
        try:
            return int(v) if v is not None else None
        except Exception:
            return None

    def _entity_label(self, e: Any) -> str:
        # 1) explicit label/name if present
        lbl = _safe_get(e, "label", None)
        if isinstance(lbl, str) and lbl.strip():
            return lbl.strip()

        cn = _safe_get(e, "class_name", None)
        if isinstance(cn, str) and cn.strip():
            return cn.strip()

        # 2) use type_dist if it's a tensor/logits/probs
        td = _safe_get(e, "type_dist", None)
        try:
            import torch
            if isinstance(td, torch.Tensor):
                t = td.detach()
                # allow shapes like (K,) or (1,K)
                if t.ndim == 2 and t.shape[0] == 1:
                    t = t[0]
                if t.ndim == 1 and t.numel() > 0:
                    # if these are logits, softmax is fine; if probs already, softmax won't break too badly
                    p = torch.softmax(t.float(), dim=-1)
                    cls = int(torch.argmax(p).item())
                    conf = float(p[cls].item())
                    from .coco80 import COCO80
                    name = COCO80[cls] if 0 <= cls < len(COCO80) else f"type {cls}"
                    return f"{name} ({conf:.2f})"
        except Exception:
            pass

        # 3) fallback to "type" if exists
        t = _safe_get(e, "type", None)
        if isinstance(t, str) and t.strip():
            return t.strip()
        if t is not None:
            return f"type {t}"

        return "object"

    def _entity_conf(self, e: Any) -> Optional[float]:
        p = _safe_get(e, "p", _safe_get(e, "conf", None))
        try:
            return float(p) if p is not None else None
        except Exception:
            return None

    def _name(self, erfg: Any, entity_id: Any) -> str:
        try:
            eid = int(entity_id)
        except Exception:
            return str(entity_id)

        for e in _entity_iter(erfg):
            if self._entity_id(e) == eid:
                lab = self._entity_label(e)
                return f"{lab} #{eid}"
        return f"entity {eid}"

    def _event_to_english(self, erfg: Any, ev: Any) -> str:
        kind = str(_safe_get(ev, "kind", "") or "")
        src = _safe_get(ev, "src", None)
        dst = _safe_get(ev, "dst", None)
        p = _safe_get(ev, "p", None)
        pp = ""
        try:
            if p is not None:
                pp = f" (p={float(p):.2f})"
        except Exception:
            pass

        a = self._name(erfg, src) if src is not None else "something"
        b = self._name(erfg, dst) if dst is not None else "something"

        # Map your canonical event kinds to natural language
        if kind.endswith("contact_begin"):
            return f"{a} touched {b}{pp}"
        if kind.endswith("contact_end"):
            return f"{a} stopped touching {b}{pp}"
        if kind.endswith("inside_begin"):
            return f"{a} went inside {b}{pp}"
        if kind.endswith("inside_end"):
            return f"{a} is no longer inside {b}{pp}"
        if kind.endswith("contains_begin"):
            return f"{b} now contains {a}{pp}"
        if kind.endswith("contains_end"):
            return f"{b} no longer contains {a}{pp}"
        if kind.endswith("support_begin"):
            return f"{b} is supporting {a}{pp}"
        if kind.endswith("support_end"):
            return f"{b} is no longer supporting {a}{pp}"

        # fallback
        return f"event {kind}: {a} -> {b}{pp}"

    def _scene_signature(self, erfg: Any) -> str:
        # lightweight hash: top labels + count
        ents = _entity_iter(erfg)
        labs = []
        for e in ents[:6]:
            labs.append(self._entity_label(e))
        return f"{len(ents)}|" + ",".join(labs)

    # ---------------------------
    # Recommendations (grounded)
    # ---------------------------

    def _recommendation(self, erfg: Any, intuition: Dict[str, Any]) -> Optional[str]:
        risk = intuition.get("risk", intuition.get("risk_map", {})) or {}
        stab = intuition.get("stability", {}) or {}
        slip = intuition.get("slip", {}) or {}
        cols = intuition.get("collision_pairs", []) or []

        risk_max = _max_dict(risk)
        unstable_max = _max_dict(stab)
        slip_max = _max_dict(slip)

        if risk_max < 0.25 and unstable_max < 0.35 and slip_max < 0.25:
            return None

        # Identify the most concerning entity (if dict keyed by entity id)
        top_r = _topk_dict(risk, 1)
        top_u = _topk_dict(stab, 1)
        top_s = _topk_dict(slip, 1)

        parts: List[str] = []

        if top_r and top_r[0][1] >= 0.30:
            parts.append(f"watch {self._name(erfg, top_r[0][0])}")

        if top_u and top_u[0][1] >= 0.55:
            parts.append(f"stabilize {self._name(erfg, top_u[0][0])}")

        if top_s and top_s[0][1] >= 0.35:
            parts.append(f"reduce slip around {self._name(erfg, top_s[0][0])}")

        # collision pairs format varies; keep it robust
        col_max = 0.0
        for p in cols if isinstance(cols, list) else []:
            if isinstance(p, dict):
                try:
                    col_max = max(col_max, float(p.get("p", 0.0)))
                except Exception:
                    pass
            elif isinstance(p, (list, tuple)) and len(p) >= 3:
                try:
                    col_max = max(col_max, float(p[2]))
                except Exception:
                    pass

        if col_max >= 0.35:
            parts.append("avoid a collision path")

        if not parts:
            return None

        if self.cfg.natural:
            return "Recommendation: " + ", ".join(parts) + "."
        return "Reco: " + ", ".join(parts) + "."

    # ---------------------------
    # Main step
    # ---------------------------

    def step(
        self,
        erfg: Any,
        events: Any,
        intuition: Dict[str, Any],
        goal_text: str = "",
        cscore: float = 0.0,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        now = time.time()
        if now - self._t_last < self.cfg.min_interval_s:
            return None, {"reason": "rate_limited"}

        events_l = _as_list(events)

        risk = intuition.get("risk", intuition.get("risk_map", {})) or {}
        slip = intuition.get("slip", {}) or {}
        unstable = intuition.get("stability", {}) or {}

        risk_max = _max_dict(risk)
        slip_max = _max_dict(slip)
        unstable_max = _max_dict(unstable)

        rb = _bucket(risk_max)
        sb = _bucket(slip_max)
        ub = _bucket(unstable_max)

        # newest event key
        ev_key = None
        ev_english = None
        if events_l:
            ev = events_l[-1]
            ev_key = f"{_safe_get(ev, 'kind', '')}|{_safe_get(ev, 'src', '')}|{_safe_get(ev, 'dst', '')}"
            ev_english = self._event_to_english(erfg, ev)

        event_changed = self.cfg.say_on_events and (ev_key is not None and ev_key != self._last_event_key)

        risk_changed = self.cfg.say_on_risk and (rb != self._last_risk_bucket) and (risk_max >= self.cfg.risk_thresh)
        slip_changed = self.cfg.say_on_risk and (sb != self._last_slip_bucket) and (slip_max >= self.cfg.slip_thresh)
        unstable_changed = self.cfg.say_on_risk and (ub != self._last_unstable_bucket) and (unstable_max >= self.cfg.unstable_thresh)

        cscore_changed = False
        if self._last_cscore is None:
            self._last_cscore = float(cscore)
        else:
            if abs(float(cscore) - float(self._last_cscore)) >= self.cfg.cscore_delta_say:
                cscore_changed = True

        # occasional scene summary (not spammy)
        scene_sig = self._scene_signature(erfg)
        scene_changed = scene_sig != self._last_scene_hash
        can_scene = now >= self._scene_cooldown

        reco = self._recommendation(erfg, intuition) if self.cfg.say_on_reco else None

        goal_active = bool((goal_text or "").strip())
        should_goal = self.cfg.say_on_goal and goal_active and cscore_changed

        should_speak = event_changed or risk_changed or slip_changed or unstable_changed or should_goal or (scene_changed and can_scene) or (reco is not None)

        debug = {
            "event_changed": event_changed,
            "risk_changed": risk_changed,
            "slip_changed": slip_changed,
            "unstable_changed": unstable_changed,
            "goal_changed": should_goal,
            "scene_changed": scene_changed and can_scene,
            "risk_max": risk_max,
            "slip_max": slip_max,
            "unstable_max": unstable_max,
            "cscore": float(cscore),
        }

        if not should_speak:
            return None, {"reason": "no_salient_change", **debug}

        sents: List[str] = []

        # 1) scene summary occasionally
        if scene_changed and can_scene:
            ents = _entity_iter(erfg)
            if ents:
                # grab a few top-names
                names = []
                for e in ents[:3]:
                    eid = self._entity_id(e)
                    if eid is None:
                        continue
                    names.append(self._name(erfg, eid))
                if names:
                    if self.cfg.natural:
                        sents.append(f"I’m tracking {len(ents)} things. Most salient: {', '.join(names)}.")
                    else:
                        sents.append(f"Tracking {len(ents)} entities: {', '.join(names)}.")
            self._last_scene_hash = scene_sig
            self._scene_cooldown = now + 4.0  # only re-summarize every few seconds

        # 2) event in english
        if event_changed and ev_english:
            if self.cfg.natural:
                sents.append(f"I noticed: {ev_english}.")
            else:
                sents.append(f"Event: {ev_english}.")

        # 3) intuition: interpret as meaning, not just numbers
        if risk_changed:
            sents.append(f"Overall risk is {_human_level(risk_max)} ({risk_max:.2f}).")
        if unstable_changed:
            sents.append(f"Stability concern is {_human_level(unstable_max)} ({unstable_max:.2f}).")
        if slip_changed:
            sents.append(f"Slip risk is {_human_level(slip_max)} ({slip_max:.2f}).")

        # 4) goal progress
        if should_goal:
            # Keep this grounded: only mention score, not “done”
            sents.append(f"Goal progress changed. Constraint score is now {float(cscore):.2f}.")

        # 5) recommendation (grounded from intuition)
        if reco is not None:
            sents.append(reco)

        # Trim + finalize
        sents = [s.strip() for s in sents if s.strip()]
        if not sents:
            sents = ["I updated my belief state."]

        sents = sents[: self.cfg.max_sentences]
        utter = " ".join(sents)
        utter = _shorten(utter, 220)

        self._t_last = now
        self._last_event_key = ev_key
        self._last_risk_bucket = rb
        self._last_slip_bucket = sb
        self._last_unstable_bucket = ub
        self._last_cscore = float(cscore)

        return utter, debug
