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
from nwm_core.models.language.narrator import NWMNarrator, NarratorConfig
from scripts.tts_speaker import TTSSpeaker


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
    panel_w: int = 440
    alpha: float = 0.45
    score_thresh: float = 0.60
    topk: int = 10
    min_area_frac: float = 0.002
    font: int = cv2.FONT_HERSHEY_SIMPLEX


def _panel(w: int, h: int) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (18, 18, 18)
    return img


def _draw_title(panel: np.ndarray, y: int, title: str, cfg: VizCfg):
    cv2.putText(panel, title, (12, y), cfg.font, 0.64, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(panel, (12, y + 7), (panel.shape[1] - 12, y + 7), (70, 70, 70), 1)


def _draw_kv(panel: np.ndarray, x: int, y: int, k: str, v: str, cfg: VizCfg, scale: float = 0.54):
    cv2.putText(panel, k, (x, y), cfg.font, scale, (175, 175, 175), 1, cv2.LINE_AA)
    cv2.putText(panel, v, (x + 170, y), cfg.font, scale, (255, 255, 255), 1, cv2.LINE_AA)


def _mask_to_frame(mask_patch: torch.Tensor, meta, resized_hw: Tuple[int, int]) -> torch.Tensor:
    mask_rs = F.interpolate(mask_patch[None, None], size=resized_hw, mode="bilinear", align_corners=False)[0, 0]
    h0, w0 = meta.orig_hw
    mask_orig = F.interpolate(mask_rs[None, None], size=(h0, w0), mode="bilinear", align_corners=False)[0, 0]
    return mask_orig


def _pick_instances_from_inst(inst: Dict[str, torch.Tensor], cfg: VizCfg) -> List[Tuple[int, int, float]]:
    class_logits = inst["class_logits"][0]
    probs = class_logits.softmax(dim=-1)
    obj_probs = probs[:, :-1]
    scores, cls_ids = obj_probs.max(dim=-1)

    valid = inst.get("valid", None)
    if valid is not None:
        v = valid[0].bool()
        scores = scores * v.float()

    keep = scores >= cfg.score_thresh
    idx = torch.where(keep)[0]
    if idx.numel() == 0:
        return []

    scored = [(int(i), int(cls_ids[i].item()), float(scores[i].item())) for i in idx]
    scored.sort(key=lambda t: t[2], reverse=True)
    return scored[: cfg.topk]


def _overlay_instances(frame_bgr: np.ndarray, evidence: dict, cfg: VizCfg) -> np.ndarray:
    if "instances" not in evidence or "pad_meta" not in evidence:
        return frame_bgr

    inst = evidence["instances"]
    if "class_logits" not in inst or "mask_logits" not in inst:
        return frame_bgr

    meta = evidence["pad_meta"]
    H0, W0 = meta.orig_hw
    min_area = int(cfg.min_area_frac * (H0 * W0))

    picks = _pick_instances_from_inst(inst, cfg)
    if not picks:
        return frame_bgr

    class_logits = inst["class_logits"][0]
    mask_logits = inst["mask_logits"][0]

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


def _deterministic_intuition(erfg: Any, evidence: dict, events: List[Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "risk": {},
        "slip": {},
        "stability": {},
        "collision_pairs": [],
        "belief_summary": [],
        "next_event": None,
    }

    def _depth_to_mask_hw(depth: torch.Tensor, H: int, W: int) -> torch.Tensor:
        if depth is None:
            return None
        if depth.ndim == 2:
            d = depth[None, None]
        elif depth.ndim == 3:
            d = depth[None] if depth.shape[0] != 1 else depth[:, None]
        elif depth.ndim == 4:
            d = depth
        else:
            return depth

        if int(d.shape[-2]) == int(H) and int(d.shape[-1]) == int(W):
            return d[0, 0]

        d_rs = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)
        return d_rs[0, 0]

    inst = evidence.get("instances", None)
    depth = evidence.get("depth", None)
    flow = evidence.get("flow", None)

    if inst is None or "mask_logits" not in inst or "class_logits" not in inst:
        return out

    device = inst["mask_logits"].device
    mask_logits = inst["mask_logits"][0]
    class_logits = inst["class_logits"][0]
    probs = class_logits.softmax(dim=-1)
    obj_probs = probs[:, :-1]
    scores, cls_ids = obj_probs.max(dim=-1)

    Q = int(mask_logits.shape[0])

    flow_mag = None
    if isinstance(flow, torch.Tensor):
        f = flow[0]
        if f.ndim == 3 and f.shape[0] == 2:
            flow_mag = torch.sqrt((f[0] ** 2 + f[1] ** 2).clamp(min=1e-12))

    for q in range(min(Q, 40)):
        sc = float(scores[q].item())
        if sc < 0.45:
            continue

        m = (torch.sigmoid(mask_logits[q]) > 0.5)
        area = float(m.float().sum().item())

        if area <= 50:
            continue

        slip = 0.0
        if flow_mag is not None:
            mm = flow_mag[m]
            if mm.numel() > 0:
                slip = float(mm.mean().item())
                slip = max(0.0, min(1.0, slip / 8.0))

        unstable = 0.0
        if depth is not None and isinstance(depth, torch.Tensor):
            d = depth[0]
            H, W = int(m.shape[0]), int(m.shape[1])

            d = evidence.get("depth", None)
            if isinstance(d, torch.Tensor):
                d0 = d[0] if d.ndim >= 3 else d
                d0 = _depth_to_mask_hw(d0, H, W)
            else:
                d0 = None

            if d0 is None:
                dm = None
            else:
                dm = d0[m]
            if dm.numel() > 0:
                dv = float(dm.std().item())
                unstable = max(0.0, min(1.0, dv / 0.35))

        risk = max(0.15 * (1.0 - sc), 0.55 * slip, 0.55 * unstable)

        out["slip"][int(q)] = float(slip)
        out["stability"][int(q)] = float(unstable)
        out["risk"][int(q)] = float(risk)

    picks = torch.where(scores >= 0.55)[0]
    picks = picks[:18]
    if picks.numel() >= 2:
        masks = (torch.sigmoid(mask_logits[picks]) > 0.5)
        for i in range(int(picks.numel())):
            for j in range(i + 1, int(picks.numel())):
                a = masks[i]
                b = masks[j]
                inter = float((a & b).float().sum().item())
                if inter <= 0.0:
                    continue
                ua = float(a.float().sum().item())
                ub = float(b.float().sum().item())
                iou = inter / max(1.0, (ua + ub - inter))
                p = max(0.0, min(1.0, iou * 2.0))
                if p >= 0.25:
                    out["collision_pairs"].append({"a": int(picks[i].item()), "b": int(picks[j].item()), "p": float(p)})

    ev_counts: Dict[str, int] = {}
    for ev in events[-20:]:
        k = _safe_get(ev, "kind", "")
        if not k:
            continue
        ev_counts[k] = ev_counts.get(k, 0) + 1

    if ev_counts:
        topk = sorted(ev_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
        out["belief_summary"] = [f"{k} x{v}" for k, v in topk]

    if len(out["collision_pairs"]) > 0:
        best = max(out["collision_pairs"], key=lambda d: float(d.get("p", 0.0)))
        out["next_event"] = f"collision_risk {best['a']}->{best['b']} p={best['p']:.2f}"
    elif len(out["risk"]) > 0:
        q_best = max(out["risk"].items(), key=lambda kv: float(kv[1]))[0]
        out["next_event"] = f"high_risk entity={q_best} p={out['risk'][q_best]:.2f}"

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

    intuition_mod = mods["IntuitionFields"]() if mods["IntuitionFields"] else None

    episodic = mods["EpisodicMemory"]() if mods["EpisodicMemory"] else None
    semantic = mods["SemanticMemory"]() if mods["SemanticMemory"] else None
    spatial = mods["SpatialMemory"]() if mods["SpatialMemory"] else None
    rulemem = mods["RuleMemory"]() if mods["RuleMemory"] else None
    mc = mods.get("MemoryConsolidator", None)

    if callable(mc) and episodic and semantic and spatial and rulemem:
        consolidator = mc(episodic, semantic, spatial, rulemem)
    else:
        consolidator = None

    mpc = mods["MicroMPC"]() if mods["MicroMPC"] else None
    prober = mods["ProbingPolicy"]() if mods["ProbingPolicy"] else None
    distill = mods["PolicyDistill"]() if mods["PolicyDistill"] else None

    goal_parser = mods["GoalParser"]() if mods["GoalParser"] else None
    mapper = mods["ConstraintMapper"]() if mods["ConstraintMapper"] else None

    narrator = NWMNarrator(NarratorConfig())
    speaker = TTSSpeaker(enabled=True)
    last_utter = ""

    bus = _CommandBus()
    threading.Thread(target=_stdin_thread, args=(bus,), daemon=True).start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (VideoCapture(0))")

    cfg = VizCfg()
    MIN_DASH_H = 900
    t_prev = time.time()
    prev: Optional[torch.Tensor] = None
    fps_ema = None
    st = LiveState()

    print("NWM Live Showcase")
    print("Type a goal into the console and press Enter:")
    print("  place the phone on the table within 3 seconds risk <= 20%")
    print("  avoid contact with the phone")
    print("  keep the cup stable and avoid collision")
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
            events = _as_list(world.get("events", []))

            mech_active = None
            mech_out_events: List[Any] = []
            if mech_router is not None and mech_exec is not None:
                try:
                    mech_active = mech_router.route(erfg, events, evidence)
                except Exception:
                    mech_active = None
                try:
                    mres = mech_exec.step(erfg, mech_active, dt, evidence=evidence)
                    if isinstance(mres, dict):
                        erfg = mres.get("erfg", erfg)
                        mech_out_events = _as_list(mres.get("events", []))
                except Exception:
                    pass

            if mech_out_events:
                events = events + mech_out_events

            intuition_out: Dict[str, Any] = {}
            if intuition_mod is not None:
                try:
                    intuition_out = intuition_mod.step(erfg, dt, events=events, evidence=evidence)
                    if intuition_out is None:
                        intuition_out = {}
                except Exception:
                    intuition_out = {}

            det_intuition = _deterministic_intuition(erfg, evidence, events)
            if not intuition_out:
                intuition_out = det_intuition
            else:
                for k in ["risk", "slip", "stability", "collision_pairs"]:
                    if k not in intuition_out or intuition_out[k] is None:
                        intuition_out[k] = det_intuition.get(k, {})
                if "belief_summary" not in intuition_out:
                    intuition_out["belief_summary"] = det_intuition.get("belief_summary", [])
                if "next_event" not in intuition_out:
                    intuition_out["next_event"] = det_intuition.get("next_event", None)

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

            utter, narr_dbg = narrator.step(
                erfg=erfg,
                events=events,
                intuition=intuition_out,
                goal_text=st.goal_text,
                cscore=st.last_constraints_score,
            )

            if utter:
                last_utter = utter
                speaker.say(utter)

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
            dash_h = max(H, MIN_DASH_H)
            panel = _panel(int(cfg.panel_w * 1.6), dash_h)

            _draw_title(panel, 24, "NWM Dashboard", cfg)
            _draw_kv(panel, 12, 56, "FPS", f"{fps_ema:.1f}" if fps_ema is not None else "?", cfg)
            _draw_kv(panel, 12, 78, "Time", f"{float(_safe_get(erfg, 'time', 0.0)):.2f}s", cfg)
            _draw_kv(panel, 12, 100, "Entities", str(len(_entity_iter(erfg))), cfg)
            _draw_kv(panel, 12, 122, "Events", str(len(events)), cfg)

            y = 154
            _draw_title(panel, y, "Goal", cfg)
            y += 30
            gt = st.goal_text if st.goal_text else "(type goal in console)"
            gt = gt[:56] + ("..." if len(gt) > 56 else "")
            cv2.putText(panel, gt, (12, y), cfg.font, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            y += 22
            _draw_kv(panel, 12, y, "CScore", f"{st.last_constraints_score:.3f}", cfg)
            y += 30

            _draw_title(panel, y, "Thoughts", cfg)
            y += 28
            txt = last_utter if last_utter else "(no narration yet)"
            line1 = txt[:54]
            line2 = txt[54:108].strip()
            cv2.putText(panel, line1, (10, y), cfg.font, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
            y += 18
            if line2:
                cv2.putText(panel, line2 + ("..." if len(txt) > 108 else ""), (10, y), cfg.font, 0.50, (255, 255, 255),
                            1, cv2.LINE_AA)
                y += 18
            y += 12

            _draw_title(panel, y, "Intuition", cfg)
            y += 30
            risk = intuition_out.get("risk", {}) or {}
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

            _draw_kv(panel, 12, y, "RiskMax", f"{max_risk:.2f}", cfg)
            y += 22
            _draw_kv(panel, 12, y, "SlipMax", f"{max_slip:.2f}", cfg)
            y += 22
            _draw_kv(panel, 12, y, "Unstable", f"{max_unstable:.2f}", cfg)
            y += 22
            _draw_kv(panel, 12, y, "Collide", f"{col_max:.2f}", cfg)
            y += 30

            _draw_title(panel, y, "Belief Summary", cfg)
            y += 30
            bs = intuition_out.get("belief_summary", []) or []
            if bs:
                for s in bs[:3]:
                    cv2.putText(panel, str(s)[:56], (12, y), cfg.font, 0.50, (230, 230, 230), 1, cv2.LINE_AA)
                    y += 18
            else:
                cv2.putText(panel, "(no summary yet)", (12, y), cfg.font, 0.50, (140, 140, 140), 1, cv2.LINE_AA)
                y += 18

            ne = intuition_out.get("next_event", None)
            if ne:
                cv2.putText(panel, ("next: " + str(ne))[:56], (12, y), cfg.font, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
                y += 20
            else:
                y += 10

            _draw_title(panel, y, "Top Entities", cfg)
            y += 30
            ents = _entity_iter(erfg)
            for e in ents[:6]:
                eid, cls, conf = _fmt_entity(e)
                cv2.putText(panel, f"#{eid}  {cls}  p={conf}", (12, y), cfg.font, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
                y += 18
            y += 10

            _draw_title(panel, y, "Recent Events", cfg)
            y += 30
            for ev in events[-6:]:
                s = _fmt_event(ev)
                s = s[:56] + ("..." if len(s) > 56 else "")
                cv2.putText(panel, s, (12, y), cfg.font, 0.50, (220, 220, 220), 1, cv2.LINE_AA)
                y += 18
            y += 10

            _draw_title(panel, y, "Plan / Action / Probe", cfg)
            y += 30
            if st.last_program_text:
                lines = st.last_program_text.splitlines()[:4]
                for ln in lines:
                    ln = ln[:56] + ("..." if len(ln) > 56 else "")
                    cv2.putText(panel, ln, (12, y), cfg.font, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
                    y += 16
            else:
                cv2.putText(panel, "(no program)", (12, y), cfg.font, 0.50, (140, 140, 140), 1, cv2.LINE_AA)
                y += 18

            a = st.last_action or {}
            p = st.last_probe or {}
            if a:
                s = ("action: " + str(a))[:56]
                cv2.putText(panel, s, (12, y), cfg.font, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
                y += 18
            else:
                cv2.putText(panel, "action: (none)", (12, y), cfg.font, 0.48, (140, 140, 140), 1, cv2.LINE_AA)
                y += 18

            if p:
                s = ("probe:  " + str(p))[:56]
                cv2.putText(panel, s, (12, y), cfg.font, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
                y += 18
            else:
                cv2.putText(panel, "probe:  (none)", (12, y), cfg.font, 0.48, (140, 140, 140), 1, cv2.LINE_AA)
                y += 18

            if vis.shape[0] < dash_h:
                pad = dash_h - vis.shape[0]
                vis = cv2.copyMakeBorder(
                    vis,
                    top=0,
                    bottom=pad,
                    left=0,
                    right=0,
                    borderType=cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                )

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
