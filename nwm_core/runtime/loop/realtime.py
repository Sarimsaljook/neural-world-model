from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# TODO: fix some of these imports
from ...common.logging import get_logger
from ...common.types import BBox
from ...models.erfg.frames import identity_frame
from ...models.erfg.io import erfg_to_json
from ...models.erfg.state import (
    Affordances,
    EntityNode,
    ERFGState,
    EventToken,
    Gaussian,
    GeometryProxy,
    HypothesisComponent,
    PhysicalPropsBelief,
    RelationEdge,
    RelationParams,
    SE3Belief,
    VelocityBelief,
)
from ...models.compiler.assoc import AssocConfig, greedy_iou_match
from ...models.compiler.filters import BBoxTrack
from ...models.compiler.relation_infer import RelationConfig, infer_relations
from ...models.intuition.fields import IntuitionConfig, compute_intuition_fields
from ...models.planning.probing import ProbingConfig, propose_probes
from ...models.memory.episodic import EpisodicMemory
from ...models.memory.semantic import SemanticMemory
from ...models.memory.rule_memory import RuleMemory
from ...models.memory.consolidation import consolidate_from_relations

from ..ingest.camera import CameraStream, CameraConfig
from ..ingest.sync import TimeSync
from .scheduler import RuntimeClocks

log = get_logger("nwm.runtime")


@dataclass
class RealtimeConfig:
    camera: CameraConfig
    clocks: RuntimeClocks
    motion_min_area: int = 700
    motion_history: int = 200
    motion_thresh: int = 35
    blur_ksize: int = 5
    max_dets: int = 12

    assoc: AssocConfig = AssocConfig()
    rel: RelationConfig = RelationConfig()
    intuition: IntuitionConfig = IntuitionConfig()
    probing: ProbingConfig = ProbingConfig()

    memory_root: str = "./data/memory"
    enable_memory: bool = True


class RealtimeEngine:
    def __init__(self, cfg: RealtimeConfig) -> None:
        self.cfg = cfg
        self.cam = CameraStream(cfg.camera)
        self.ts = TimeSync()
        self.sched = cfg.clocks.build()

        self._bg = cv2.createBackgroundSubtractorMOG2(history=cfg.motion_history, varThreshold=cfg.motion_thresh, detectShadows=False)

        self._next_id = 1
        self._tracks: Dict[str, BBoxTrack] = {}
        self._attach_counters: Dict[Tuple[str, str], int] = {}
        self._prev_rel_set: set[Tuple[str, str, str]] = set()
        self._prev_visible: Dict[str, bool] = {}

        self._events: List[EventToken] = []
        self._intuition: Dict[str, Dict[str, float]] = {}
        self._uncertainty: Dict[str, float] = {}

        self._state = self._init_state()

        if cfg.enable_memory:
            root = cv2.os.path.join(cfg.memory_root)
            self._episodic = EpisodicMemory(root=cv2.os.path.join(cfg.memory_root) and __import__("pathlib").Path(cfg.memory_root) / "episodic")
            self._semantic = SemanticMemory(path=__import__("pathlib").Path(cfg.memory_root) / "semantic.json")
            self._rules = RuleMemory(path=__import__("pathlib").Path(cfg.memory_root) / "rules.json")
        else:
            self._episodic = None
            self._semantic = None
            self._rules = None

    def _init_state(self) -> ERFGState:
        world = identity_frame("world")
        ego = identity_frame("ego")
        hyp = HypothesisComponent(weight=1.0, entities={}, relations=[], world_frame=world)
        return ERFGState(
            timestamp_ns=self.ts.now_ns(),
            ego_frame=ego,
            world_frame=world,
            hypotheses=[hyp],
            active_entities=[],
            version=0,
        )

    def close(self) -> None:
        self.cam.close()

    def export_state(self) -> Dict[str, Any]:
        s = erfg_to_json(self._state)
        s["events"] = [e.__dict__ for e in self._events[-50:]]
        s["intuition"] = self._intuition
        s["uncertainty"] = self._uncertainty
        return s

    def get_state(self) -> ERFGState:
        return self._state

    def propose_probes(self):
        return propose_probes(self._state, self._uncertainty, self._intuition, self.cfg.probing)

    def step(self) -> Optional[np.ndarray]:
        frame = self.cam.read()
        if frame is None:
            return None

        now_ns = self.ts.now_ns()
        self._state.timestamp_ns = now_ns
        self._state.version += 1

        dets = self._detect_motion_objects(frame)
        self._update_tracks(dets)
        rels = self._compute_relations()
        self._events.extend(self._detect_events(rels))
        self._intuition = compute_intuition_fields(self._track_boxes(), self._track_vels(), rels, self.cfg.intuition)

        self._update_erfg(rels)

        # episodic memory (low rate)
        if self.cfg.enable_memory and self._episodic is not None:
            # save ~1Hz
            if self.sched.should_tick("event"):
                day = time.strftime("%Y%m%d", time.localtime())
                self._episodic.append(day, self._state, extras={"events": [e.__dict__ for e in self._events[-10:]]})

        return frame

    # ---------------------------
    # Prediction & counterfactuals
    # ---------------------------
    def predict(self, horizon: int = 15, dt: float = 1.0) -> Dict[str, Any]:
        boxes = self._track_boxes()
        vels = self._track_vels()
        preds: Dict[str, List[Dict[str, float]]] = {}
        for eid, b in boxes.items():
            v = vels.get(eid, np.zeros(2))
            seq = []
            for k in range(1, horizon + 1):
                dx, dy = float(v[0] * dt * k), float(v[1] * dt * k)
                seq.append({"x1": b.x1 + dx, "y1": b.y1 + dy, "x2": b.x2 + dx, "y2": b.y2 + dy})
            preds[eid] = seq
        return {"horizon": horizon, "dt": dt, "pred_boxes": preds}

    def counterfactual(self, horizon: int, dt: float, interventions: Dict[str, Any]) -> Dict[str, Any]:
        boxes = dict(self._track_boxes())
        vels = dict(self._track_vels())

        remove = set(interventions.get("remove", []) or [])
        vel_delta = interventions.get("velocity_delta", {}) or {}

        for rid in remove:
            boxes.pop(rid, None)
            vels.pop(rid, None)

        for eid, dxy in vel_delta.items():
            if eid in vels:
                dxy = np.asarray(dxy, dtype=np.float64).reshape(2)
                vels[eid] = vels[eid] + dxy

        preds: Dict[str, List[Dict[str, float]]] = {}
        for eid, b in boxes.items():
            v = vels.get(eid, np.zeros(2))
            seq = []
            for k in range(1, horizon + 1):
                dx, dy = float(v[0] * dt * k), float(v[1] * dt * k)
                seq.append({"x1": b.x1 + dx, "y1": b.y1 + dy, "x2": b.x2 + dx, "y2": b.y2 + dy})
            preds[eid] = seq

        return {"horizon": horizon, "dt": dt, "interventions": interventions, "pred_boxes": preds}

    def consolidate_memory(self) -> None:
        if not (self.cfg.enable_memory and self._semantic and self._rules):
            return
        rels = self._current_relations()
        consolidate_from_relations(self._semantic, self._rules, rels)
        log.info("Memory consolidated: semantic + rule priors updated")

    def _detect_motion_objects(self, frame_rgb: np.ndarray) -> List[BBox]:
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        if self.cfg.blur_ksize > 1:
            gray = cv2.GaussianBlur(gray, (self.cfg.blur_ksize, self.cfg.blur_ksize), 0)

        fg = self._bg.apply(gray)
        fg = cv2.medianBlur(fg, 5)
        fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)[1]

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: List[BBox] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.cfg.motion_min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            boxes.append(BBox(float(x), float(y), float(x + w), float(y + h)))

        boxes.sort(key=lambda b: -b.area())
        return boxes[: self.cfg.max_dets]

    def _update_tracks(self, dets: List[BBox]) -> None:
        # predict existing
        for tr in self._tracks.values():
            tr.predict(dt=1.0)

        trk_ids = list(self._tracks.keys())
        trk_boxes = [self._tracks[k].bbox for k in trk_ids]
        matches = greedy_iou_match(dets, trk_boxes, min_iou=self.cfg.assoc.min_iou)

        matched_trks = set()
        matched_dets = set()

        for di, tj, _ in matches:
            det = dets[di]
            tid = trk_ids[tj]
            self._tracks[tid].update(det)
            matched_trks.add(tid)
            matched_dets.add(di)

        # age unmatched
        for tid, tr in list(self._tracks.items()):
            if tid not in matched_trks:
                tr.time_since_update += 1
                if tr.time_since_update > self.cfg.assoc.max_age:
                    del self._tracks[tid]

        # new tracks
        for i, det in enumerate(dets):
            if i in matched_dets:
                continue
            tid = f"e{self._next_id}"
            self._next_id += 1
            self._tracks[tid] = BBoxTrack(track_id=self._next_id, bbox=det, v_xy=np.zeros(2, dtype=np.float64))
            self._tracks[tid].hits = 1

        # update uncertainty map
        self._uncertainty = {tid: tr.uncertainty() for tid, tr in self._tracks.items()}

    def _track_boxes(self) -> Dict[str, BBox]:
        # only confirmed
        out: Dict[str, BBox] = {}
        for tid, tr in self._tracks.items():
            if tr.hits >= self.cfg.assoc.min_hits:
                out[tid] = tr.bbox
        return out

    def _track_vels(self) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for tid, tr in self._tracks.items():
            if tr.hits >= self.cfg.assoc.min_hits:
                out[tid] = tr.v_xy.copy()
        return out

    def _compute_relations(self) -> List[Dict]:
        boxes = self._track_boxes()
        vels = self._track_vels()
        rels, self._attach_counters = infer_relations(boxes, vels, self._attach_counters, self.cfg.rel)
        self._rels = rels
        return rels

    def _current_relations(self) -> List[Dict]:
        return getattr(self, "_rels", [])

    def _detect_events(self, rels: List[Dict]) -> List[EventToken]:
        now_ns = self._state.timestamp_ns
        rel_set = {(r["src"], r["dst"], r["type"]) for r in rels}

        events: List[EventToken] = []

        # relation transitions
        added = rel_set - self._prev_rel_set
        removed = self._prev_rel_set - rel_set

        for (s, d, t) in added:
            et = None
            if t == "contact":
                et = "contact_begin"
            elif t == "supporting":
                et = "support_gained"
            elif t == "attached":
                et = "attach_inferred"
            elif t == "inside":
                et = "occlusion_enter"
            if et:
                events.append(EventToken(event_type=et, timestamp_ns=now_ns, participants=[s, d], confidence=0.75, params={"type": t}))

        for (s, d, t) in removed:
            et = None
            if t == "contact":
                et = "contact_end"
            elif t == "supporting":
                et = "support_lost"
            elif t == "attached":
                et = "detach_inferred"
            elif t == "inside":
                et = "occlusion_exit"
            if et:
                events.append(EventToken(event_type=et, timestamp_ns=now_ns, participants=[s, d], confidence=0.7, params={"type": t}))

        self._prev_rel_set = rel_set
        return events

    def _update_erfg(self, rels: List[Dict]) -> None:
        hyp = self._state.hypotheses[0]
        hyp.entities.clear()
        hyp.relations.clear()

        boxes = self._track_boxes()
        vels = self._track_vels()

        active = []
        for eid, b in boxes.items():
            active.append(eid)

            cx = (b.x1 + b.x2) * 0.5
            cy = (b.y1 + b.y2) * 0.5
            pos_mean = np.array([cx, cy, 1.0], dtype=np.float64)
            pos_cov = np.diag([max(4.0, self._tracks[eid].pos_var[0]), max(4.0, self._tracks[eid].pos_var[1]), 25.0]).astype(np.float64)

            rot_mean = np.zeros(3, dtype=np.float64)
            rot_cov = np.diag([0.5, 0.5, 0.5]).astype(np.float64)

            lin_mean = np.array([float(vels.get(eid, np.zeros(2))[0]), float(vels.get(eid, np.zeros(2))[1]), 0.0], dtype=np.float64)
            lin_cov = np.diag([max(2.0, self._tracks[eid].vel_var[0]), max(2.0, self._tracks[eid].vel_var[1]), 9.0]).astype(np.float64)

            ang_mean = np.zeros(3, dtype=np.float64)
            ang_cov = np.diag([1.0, 1.0, 1.0]).astype(np.float64)

            pose = SE3Belief(position=Gaussian(pos_mean, pos_cov), rotation=Gaussian(rot_mean, rot_cov))
            vel = VelocityBelief(linear=Gaussian(lin_mean, lin_cov), angular=Gaussian(ang_mean, ang_cov))

            # physical props beliefs
            props = PhysicalPropsBelief(
                mass=Gaussian(np.array([1.0], dtype=np.float64), np.array([[4.0]], dtype=np.float64)),
                friction=Gaussian(np.array([0.5], dtype=np.float64), np.array([[0.2]], dtype=np.float64)),
                restitution=Gaussian(np.array([0.2], dtype=np.float64), np.array([[0.2]], dtype=np.float64)),
                stiffness=Gaussian(np.array([0.5], dtype=np.float64), np.array([[0.5]], dtype=np.float64)),
                damping=Gaussian(np.array([0.5], dtype=np.float64), np.array([[0.5]], dtype=np.float64)),
            )

            w = max(1.0, b.x2 - b.x1)
            h = max(1.0, b.y2 - b.y1)
            size = float(np.clip((w * h) / (640.0 * 480.0), 0.0, 1.0))
            aff = Affordances(
                graspable=float(np.clip(1.0 - size, 0.0, 1.0)),
                pushable=0.8,
                pullable=0.4,
                openable=0.2,
                pourable=0.1,
                fragile=0.2,
                hot=0.0,
                sharp=0.0,
                support_surface=float(np.clip(size * 1.2, 0.0, 1.0)),
                container=0.1,
            )

            geom = GeometryProxy(kind="bbox", params={"xyxy": np.array([b.x1, b.y1, b.x2, b.y2], dtype=np.float64)})

            app = np.zeros(64, dtype=np.float64)  # placeholder appearance embedding; can be replaced with encoder features
            hyp.entities[eid] = EntityNode(
                entity_id=eid,
                type_logits=np.zeros(16, dtype=np.float64),
                pose=pose,
                velocity=vel,
                geometry=geom,
                props=props,
                affordances=aff,
                appearance_embed=app,
                last_seen_ts=int(self._state.timestamp_ns),
                alive_prob=1.0,
                parts={},
                extras={"bbox_xyxy": geom.params["xyxy"].tolist()},
            )

        for r in rels:
            src, dst, typ = r["src"], r["dst"], r["type"]
            conf = float(r.get("confidence", 0.5))
            # predicate logits small fixed vector, with the predicted type highest
            pred_vocab = ["contact", "supporting", "supported_by", "inside", "contains", "attached"]
            logits = np.full((len(pred_vocab),), -2.0, dtype=np.float64)
            if typ in pred_vocab:
                logits[pred_vocab.index(typ)] = 2.0
            hyp.relations.append(
                RelationEdge(
                    src=src,
                    dst=dst,
                    predicate_logits=logits,
                    predicate=typ if typ in pred_vocab else None,
                    confidence=conf,
                    params=RelationParams(params={}),
                    extras={},
                )
            )

        self._state.active_entities = active

        # global world state uncertainty for runtime gating and probing
        if active:
            self._uncertainty["_world"] = float(np.mean([self._uncertainty[e] for e in active]))
        else:
            self._uncertainty["_world"] = 0.0
