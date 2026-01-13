from __future__ import annotations

import time
import threading
import queue
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from nwm_core.models.encoder.encoder_model import EvidenceEncoder
from nwm_core.models.compiler import WorldCompiler, CompilerConfig


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


def preprocess(frame_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    return x


@dataclass
class VizCfg:
    topk: int = 10
    score_thresh: float = 0.60
    min_area_frac: float = 0.002
    alpha: float = 0.45
    panel_w: int = 420
    font: int = cv2.FONT_HERSHEY_SIMPLEX


def _pick_instances(class_logits: torch.Tensor, score_thresh: float, topk: int) -> List[Tuple[int, int, float]]:
    probs = class_logits.softmax(dim=-1)
    obj_probs = probs[:, :-1]
    scores, cls_ids = obj_probs.max(dim=-1)
    keep = scores >= score_thresh
    idx = torch.where(keep)[0]
    if idx.numel() == 0:
        return []
    scored = [(int(i), int(cls_ids[i].item()), float(scores[i].item())) for i in idx]
    scored.sort(key=lambda t: t[2], reverse=True)
    return scored[:topk]


def _mask_to_frame(mask_patch: torch.Tensor, meta, resized_hw: Tuple[int, int]) -> torch.Tensor:
    mask_rs = F.interpolate(mask_patch[None, None], size=resized_hw, mode="bilinear", align_corners=False)[0, 0]
    h0, w0 = meta.orig_hw
    mask_orig = F.interpolate(mask_rs[None, None], size=(h0, w0), mode="bilinear", align_corners=False)[0, 0]
    return mask_orig


def _overlay_instances(frame_bgr: np.ndarray, out: dict, cfg: VizCfg) -> np.ndarray:
    inst = out["instances"]
    class_logits = inst["class_logits"][0]
    mask_logits = inst["mask_logits"][0]
    meta = out["pad_meta"]

    H0, W0 = meta.orig_hw
    min_area = int(cfg.min_area_frac * (H0 * W0))

    picks = _pick_instances(class_logits, cfg.score_thresh, cfg.topk)
    if not picks:
        return frame_bgr

    overlay = frame_bgr.copy()

    for (q, cls_id, score) in picks:
        mask_patch = torch.sigmoid(mask_logits[q])
        mask = _mask_to_frame(mask_patch, meta, resized_hw=meta.resized_hw)

        bin_mask = (mask > 0.5).detach().cpu().numpy().astype(np.uint8)
        area = int(bin_mask.sum())
        if area < min_area:
            continue

        ys, xs = np.where(bin_mask > 0)
        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())

        rng = np.random.RandomState(cls_id * 9973 + 17)
        color = tuple(int(x) for x in rng.randint(60, 255, size=3))

        colored = np.zeros_like(frame_bgr, dtype=np.uint8)
        colored[:, :] = color
        overlay = np.where(
            bin_mask[..., None] > 0,
            (cfg.alpha * colored + (1 - cfg.alpha) * overlay).astype(np.uint8),
            overlay,
        )

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            overlay,
            f"cls={cls_id} p={score:.2f}",
            (x1, max(0, y1 - 6)),
            cfg.font,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return overlay


def _panel(w: int, h: int) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (20, 20, 20)
    return img


def _draw_kv(panel: np.ndarray, x: int, y: int, k: str, v: str, cfg: VizCfg, scale: float = 0.55):
    cv2.putText(panel, k, (x, y), cfg.font, scale, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(panel, v, (x + 140, y), cfg.font, scale, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_title(panel: np.ndarray, y: int, title: str, cfg: VizCfg):
    cv2.putText(panel, title, (10, y), cfg.font, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(panel, (10, y + 6), (panel.shape[1] - 10, y + 6), (70, 70, 70), 1)


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


def _event_iter(events: Any) -> List[Any]:
    return _as_list(events)


def _fmt_entity(e: Any) -> Tuple[str, str, str]:
    eid = _safe_get(e, "id", _safe_get(e, "entity_id", None))
    cls = _safe_get(e, "class_name", _safe_get(e, "cls", _safe_get(e, "type", None)))
    p = _safe_get(e, "p", _safe_get(e, "conf", None))
    return str(eid if eid is not None else "?"), str(cls if cls is not None else "?"), f"{float(p):.2f}" if p is not None else "?"


def _fmt_event(ev: Any) -> str:
    k = _safe_get(ev, "kind", "")
    src = _safe_get(ev, "src", None)
    dst = _safe_get(ev, "dst", None)
    p = _safe_get(ev, "p", None)
    if p is None:
        return f"{k} {src}->{dst}"
    return f"{k} {src}->{dst} p={float(p):.2f}"


class _CommandBus:
    def __init__(self):
        self._q: "queue.Queue[str]" = queue.Queue()
        self._last: Optional[str] = None

    def push(self, s: str):
        s = (s or "").strip()
        if not s:
            return
        self._q.put(s)

    def poll_latest(self) -> Optional[str]:
        got = None
        while True:
            try:
                got = self._q.get_nowait()
                self._last = got
            except queue.Empty:
                break
        return got

    @property
    def last(self) -> Optional[str]:
        return self._last


def _stdin_thread(bus: _CommandBus):
    while True:
        try:
            s = input().strip()
        except EOFError:
            return
        if not s:
            continue
        bus.push(s)


def _try_import_modules() -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    try:
        from nwm_core.models.mechanisms.router import MechanismRouter
        out["MechanismRouter"] = MechanismRouter
    except Exception:
        out["MechanismRouter"] = None

    try:
        from nwm_core.models.mechanisms.executor import MechanismExecutor
        out["MechanismExecutor"] = MechanismExecutor
    except Exception:
        out["MechanismExecutor"] = None

    try:
        from nwm_core.models.intuition.fields import IntuitionFields
        out["IntuitionFields"] = IntuitionFields
    except Exception:
        out["IntuitionFields"] = None

    try:
        from nwm_core.models.memory.episodic import EpisodicMemory
        from nwm_core.models.memory.semantic import SemanticMemory
        from nwm_core.models.memory.spatial import SpatialMemory
        from nwm_core.models.memory.rule_memory import RuleMemory
        from nwm_core.models.memory.consolidation import MemoryConsolidator
        out["EpisodicMemory"] = EpisodicMemory
        out["SemanticMemory"] = SemanticMemory
        out["SpatialMemory"] = SpatialMemory
        out["RuleMemory"] = RuleMemory
        out["MemoryConsolidator"] = MemoryConsolidator
    except Exception:
        out["EpisodicMemory"] = None
        out["SemanticMemory"] = None
        out["SpatialMemory"] = None
        out["RuleMemory"] = None
        out["MemoryConsolidator"] = None

    try:
        from nwm_core.models.planning.constraints import build_constraint_set, score_constraints
        out["build_constraint_set"] = build_constraint_set
        out["score_constraints"] = score_constraints
    except Exception:
        out["build_constraint_set"] = None
        out["score_constraints"] = None

    try:
        from nwm_core.models.planning.event_programs import synthesize_program_from_language, program_to_text
        out["synthesize_program_from_language"] = synthesize_program_from_language
        out["program_to_text"] = program_to_text
    except Exception:
        out["synthesize_program_from_language"] = None
        out["program_to_text"] = None

    try:
        from nwm_core.models.planning.mpc import MicroMPC
        out["MicroMPC"] = MicroMPC
    except Exception:
        out["MicroMPC"] = None

    try:
        from nwm_core.models.planning.probing import ProbingPolicy
        out["ProbingPolicy"] = ProbingPolicy
    except Exception:
        out["ProbingPolicy"] = None

    try:
        from nwm_core.models.planning.policy_distill import PolicyDistill
        out["PolicyDistill"] = PolicyDistill
    except Exception:
        out["PolicyDistill"] = None

    try:
        from nwm_core.models.language.goal_parser import GoalParser
        from nwm_core.models.language.constraint_mapper import ConstraintMapper
        out["GoalParser"] = GoalParser
        out["ConstraintMapper"] = ConstraintMapper
    except Exception:
        out["GoalParser"] = None
        out["ConstraintMapper"] = None

    return out


@dataclass
class LiveState:
    goal_text: str = ""
    mapped_goal: Optional[Dict[str, Any]] = None
    last_program_text: str = ""
    last_action: Optional[Dict[str, Any]] = None
    last_probe: Optional[Dict[str, Any]] = None
    last_constraints_score: float = 0.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mods = _try_import_modules()

    encoder = EvidenceEncoder(num_classes=80, use_keypoints=False).to(device).eval()
    compiler = WorldCompiler(CompilerConfig(), device=device)

    mech_router = mods["MechanismRouter"]() if mods["MechanismRouter"] else None
    mech_exec = mods["MechanismExecutor"]() if mods["MechanismExecutor"] else None

    intuition = mods["IntuitionFields"]() if mods["IntuitionFields"] else None

    episodic = mods["EpisodicMemory"]() if mods["EpisodicMemory"] else None
    semantic = mods["SemanticMemory"]() if mods["SemanticMemory"] else None
    spatial = mods["SpatialMemory"]() if mods["SpatialMemory"] else None
    rulemem = mods["RuleMemory"]() if mods["RuleMemory"] else None
    consolidator = mods["MemoryConsolidator"]() if mods["MemoryConsolidator"] else None

    mpc = mods["MicroMPC"]() if mods["MicroMPC"] else None
    prober = mods["ProbingPolicy"]() if mods["ProbingPolicy"] else None
    distill = mods["PolicyDistill"]() if mods["PolicyDistill"] else None

    goal_parser = mods["GoalParser"]() if mods["GoalParser"] else None
    mapper = mods["ConstraintMapper"]() if mods["ConstraintMapper"] else None

    bus = _CommandBus()
    threading.Thread(target=_stdin_thread, args=(bus,), daemon=True).start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (VideoCapture(0))")

    cfg = VizCfg()
    t_prev = time.time()
    prev: Optional[torch.Tensor] = None
    fps_ema = None

    st = LiveState()

    print("NWM Live Demo")
    print("Type a goal into the console and press Enter (examples):")
    print("  place the cup on the table within 3 seconds risk <= 20%")
    print("  avoid contact with the phone")
    print("Press 'q' in the window to quit.")

    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.time()
            dt = now - t_prev
            t_prev = now

            if dt > 1e-6:
                fps = 1.0 / dt
                fps_ema = fps if fps_ema is None else (0.9 * fps_ema + 0.1 * fps)

            cmd = bus.poll_latest()
            if cmd is not None and goal_parser is not None and mapper is not None:
                st.goal_text = cmd
                goal_spec = goal_parser.parse(cmd)
                st.mapped_goal = mapper.map(goal_spec)

            x = preprocess(frame, device)
            evidence = encoder(x, prev)

            world = compiler.step(evidence, dt)
            erfg = world["erfg"]
            events = _event_iter(world["events"])

            mech_active = None
            mech_out_events: List[Any] = []
            if mech_router is not None and mech_exec is not None:
                try:
                    mech_active = mech_router.route(erfg, events, evidence)  # expected: list/dict
                except Exception:
                    mech_active = None
                try:
                    mres = mech_exec.step(erfg, mech_active, dt, evidence=evidence)
                    if isinstance(mres, dict):
                        erfg = mres.get("erfg", erfg)
                        mech_out_events = _event_iter(mres.get("events", []))
                except Exception:
                    pass

            if mech_out_events:
                events = events + mech_out_events

            intuition_out: Dict[str, Any] = {}
            if intuition is not None:
                try:
                    intuition_out = intuition.step(erfg, dt, events=events, evidence=evidence)
                    if intuition_out is None:
                        intuition_out = {}
                except Exception:
                    intuition_out = {}

            if episodic is not None:
                try:
                    episodic.update(erfg, events, intuition_out)
                except Exception:
                    pass

            if semantic is not None:
                try:
                    semantic.update(erfg, events, intuition_out)
                except Exception:
                    pass

            if spatial is not None:
                try:
                    spatial.update(erfg, events, intuition_out)
                except Exception:
                    pass

            if rulemem is not None:
                try:
                    rulemem.update(erfg, events, intuition_out)
                except Exception:
                    pass

            if consolidator is not None:
                try:
                    consolidator.step(episodic=episodic, semantic=semantic, spatial=spatial, rulemem=rulemem)
                except Exception:
                    pass

            st.last_program_text = ""
            st.last_action = None
            st.last_probe = None
            st.last_constraints_score = 0.0

            if st.mapped_goal is not None:
                if mods["build_constraint_set"] is not None and mods["score_constraints"] is not None:
                    try:
                        cset = mods["build_constraint_set"](
                            parsed_goal={"risk_budget": float(st.mapped_goal.get("goal", {}).get("risk_budget", 0.25))},
                            mapped_constraints=st.mapped_goal.get("constraints", []),
                            mapped_avoid=st.mapped_goal.get("avoid", []),
                        )
                        st.last_constraints_score = float(mods["score_constraints"](cset, intuition_out, erfg))
                    except Exception:
                        st.last_constraints_score = 0.0

                if mods["synthesize_program_from_language"] is not None and mods["program_to_text"] is not None:
                    try:
                        prog = mods["synthesize_program_from_language"](erfg, events, st.mapped_goal)
                        st.last_program_text = mods["program_to_text"](prog)
                    except Exception:
                        st.last_program_text = ""

                if mpc is not None:
                    try:
                        st.last_action = mpc.propose_action(
                            erfg,
                            st.last_constraints_score,
                            events,
                            intuition=intuition_out,
                            goal=st.mapped_goal.get("goal", None),
                        )
                    except Exception:
                        st.last_action = None

                if prober is not None:
                    try:
                        st.last_probe = prober.suggest(erfg, intuition_out, events=events, goal=st.mapped_goal.get("goal", None))
                    except Exception:
                        st.last_probe = None

                if distill is not None:
                    try:
                        _ = distill.step(erfg, events=events, intuition=intuition_out, goal=st.mapped_goal.get("goal", None))
                    except Exception:
                        pass

            vis = _overlay_instances(frame, evidence, cfg)

            H, W = vis.shape[:2]
            panel = _panel(cfg.panel_w, H)

            _draw_title(panel, 24, "NWM Dashboard", cfg)
            _draw_kv(panel, 10, 52, "FPS", f"{fps_ema:.1f}" if fps_ema is not None else "?", cfg)
            _draw_kv(panel, 10, 74, "Time", f"{float(_safe_get(erfg, 'time', 0.0)):.2f}s", cfg)
            _draw_kv(panel, 10, 96, "Entities", str(len(_entity_iter(erfg))), cfg)
            _draw_kv(panel, 10, 118, "Events", str(len(events)), cfg)

            y = 150
            _draw_title(panel, y, "Goal", cfg)
            y += 28
            gt = st.goal_text if st.goal_text else "(type into console)"
            gt = gt[:52] + ("..." if len(gt) > 52 else "")
            cv2.putText(panel, gt, (10, y), cfg.font, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            y += 22
            _draw_kv(panel, 10, y, "CScore", f"{st.last_constraints_score:.3f}", cfg)
            y += 28

            _draw_title(panel, y, "Intuition", cfg)
            y += 28
            risk = intuition_out.get("risk", intuition_out.get("risk_map", {})) or {}
            stab = intuition_out.get("stability", {}) or {}
            slip = intuition_out.get("slip", {}) or {}
            col = intuition_out.get("collision_pairs", []) or []

            def _max_dict(d: Dict) -> float:
                m = 0.0
                for _, v in d.items():
                    try:
                        m = max(m, float(v))
                    except Exception:
                        pass
                return float(m)

            max_risk = _max_dict(risk) if isinstance(risk, dict) else 0.0
            max_slip = _max_dict(slip) if isinstance(slip, dict) else 0.0
            max_unstable = _max_dict(stab) if isinstance(stab, dict) else 0.0

            col_max = 0.0
            if isinstance(col, list):
                for p in col:
                    if isinstance(p, dict):
                        col_max = max(col_max, float(p.get("p", 0.0)))
                    elif isinstance(p, (list, tuple)) and len(p) >= 3:
                        try:
                            col_max = max(col_max, float(p[2]))
                        except Exception:
                            pass

            _draw_kv(panel, 10, y, "RiskMax", f"{max_risk:.2f}", cfg)
            y += 22
            _draw_kv(panel, 10, y, "SlipMax", f"{max_slip:.2f}", cfg)
            y += 22
            _draw_kv(panel, 10, y, "Unstable", f"{max_unstable:.2f}", cfg)
            y += 22
            _draw_kv(panel, 10, y, "Collide", f"{col_max:.2f}", cfg)
            y += 30

            _draw_title(panel, y, "Top Entities", cfg)
            y += 28
            ents = _entity_iter(erfg)
            for e in ents[:6]:
                eid, cls, conf = _fmt_entity(e)
                cv2.putText(panel, f"#{eid}  {cls}  p={conf}", (10, y), cfg.font, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
                y += 18
            y += 14

            _draw_title(panel, y, "Recent Events", cfg)
            y += 28
            for ev in events[-6:]:
                s = _fmt_event(ev)
                s = s[:54] + ("..." if len(s) > 54 else "")
                cv2.putText(panel, s, (10, y), cfg.font, 0.50, (220, 220, 220), 1, cv2.LINE_AA)
                y += 18
            y += 14

            _draw_title(panel, y, "Plan", cfg)
            y += 28
            if st.last_program_text:
                lines = st.last_program_text.splitlines()[:6]
                for ln in lines:
                    ln = ln[:54] + ("..." if len(ln) > 54 else "")
                    cv2.putText(panel, ln, (10, y), cfg.font, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
                    y += 16
            else:
                cv2.putText(panel, "(no program)", (10, y), cfg.font, 0.50, (150, 150, 150), 1, cv2.LINE_AA)
                y += 18
            y += 10

            _draw_title(panel, y, "Action / Probe", cfg)
            y += 28
            a = st.last_action or {}
            p = st.last_probe or {}

            if a:
                s = str(a)
                s = s[:54] + ("..." if len(s) > 54 else "")
                cv2.putText(panel, "action: " + s, (10, y), cfg.font, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
                y += 18
            else:
                cv2.putText(panel, "action: (none)", (10, y), cfg.font, 0.48, (150, 150, 150), 1, cv2.LINE_AA)
                y += 18

            if p:
                s = str(p)
                s = s[:54] + ("..." if len(s) > 54 else "")
                cv2.putText(panel, "probe:  " + s, (10, y), cfg.font, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
                y += 18
            else:
                cv2.putText(panel, "probe:  (none)", (10, y), cfg.font, 0.48, (150, 150, 150), 1, cv2.LINE_AA)
                y += 18

            dash = np.concatenate([vis, panel], axis=1)

            cv2.putText(
                dash,
                "NWM Live | type goal in console | q=quit",
                (10, 24),
                cfg.font,
                0.70,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("NWM Live Demo", dash)

            prev = x

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
