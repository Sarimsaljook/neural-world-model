from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from nwm_core.models.erfg import ERFG, EntityNode, GaussianState
from .assoc import greedy_associate
from .filters import CVFilter
from .relation_infer import cxcywh_to_xyxy, infer_pair_relations


@dataclass
class CompilerConfig:
    max_entities: int = 256
    max_missed: int = 30
    det_min_conf: float = 0.25
    det_min_area: float = 0.0025
    assoc_min_score: float = 0.25
    relation_threshold: float = 0.5
    mask_thresh: float = 0.2
    det_min_conf: float = 0.05
    det_min_area: float = 0.0005

@dataclass
class EventToken:
    t: float
    kind: str
    src: int
    dst: int
    p: float


class WorldCompiler:
    def __init__(self, cfg: Optional[CompilerConfig] = None, device: Optional[torch.device] = None):
        self.cfg = cfg or CompilerConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.erfg = ERFG()
        self._next_entity_id = 0

        self._track_ids: List[int] = []
        self._track_filters: Dict[int, CVFilter] = {}
        self._track_embs: Dict[int, torch.Tensor] = {}
        self._track_boxes: Dict[int, torch.Tensor] = {}
        self._track_missed: Dict[int, int] = {}

        self._prev_rel_probs: Dict[Tuple[int, int, str], float] = {}

    def _new_entity_id(self) -> int:
        eid = self._next_entity_id
        self._next_entity_id += 1
        return eid

    @staticmethod
    def _boxes_cxcywh_to_xyxy_norm(boxes: torch.Tensor) -> torch.Tensor:
        return cxcywh_to_xyxy(boxes)

    @staticmethod
    def _mask_probs(mask_logits: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
        # mask logits
        probs = torch.sigmoid(mask_logits.unsqueeze(0))
        probs = F.interpolate(probs, size=(out_h, out_w), mode="bilinear", align_corners=False)
        return probs.squeeze(0)

    def _extract_detections(self, evidence: Dict) -> Dict[str, torch.Tensor]:
        inst = evidence["instances"]

        if "masks" in inst:
            masks = inst["masks"][0].to(self.device)
        elif "mask_logits" in inst:
            masks = torch.sigmoid(inst["mask_logits"][0]).to(self.device)
        else:
            raise KeyError("instances must contain 'masks' or 'mask_logits'")

        # class logits
        if "class_logits" not in inst:
            raise KeyError("instances must contain 'class_logits'")
        logits = inst["class_logits"][0].to(self.device)

        # query embeddings
        if "query_feats" in inst:
            embs = inst["query_feats"][0].to(self.device)
        elif "embeddings" in inst:
            embs = inst["embeddings"][0].to(self.device)
        else:
            raise KeyError("instances must contain 'query_feats' or 'embeddings'")
        embs = F.normalize(embs, dim=-1)

        if "depth" not in evidence:
            raise KeyError("evidence must contain 'depth'")
        depth_t = evidence["depth"][0].to(self.device)
        if depth_t.ndim == 3:
            depth_t = depth_t.squeeze(0)
        elif depth_t.ndim == 4:
            depth_t = depth_t.squeeze(0).squeeze(0)
        depth = depth_t

        meta = evidence.get("meta", None)
        if meta is not None:
            H, W = int(meta.orig_hw[0]), int(meta.orig_hw[1])
        else:
            H, W = int(depth.shape[-2]), int(depth.shape[-1])

        mh, mw = int(masks.shape[-2]), int(masks.shape[-1])
        if (mh, mw) != (H, W):
            masks = F.interpolate(masks.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False).squeeze(1)

        dh, dw = int(depth.shape[-2]), int(depth.shape[-1])
        if (dh, dw) != (H, W):
            depth = F.interpolate(depth.unsqueeze(0).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)

        probs = torch.softmax(logits, dim=-1)
        if probs.shape[-1] > 1:
            conf, cls = probs.max(dim=-1)
        else:
            conf = probs.squeeze(-1)
            cls = torch.zeros_like(conf, dtype=torch.long)

        # compute boxes from masks
        q = masks.shape[0]
        boxes_xyxy = torch.zeros((q, 4), device=self.device, dtype=torch.float32)
        areas = torch.zeros((q,), device=self.device, dtype=torch.float32)

        bin_masks = masks > float(self.cfg.mask_thresh)
        for i in range(q):
            ys, xs = torch.where(bin_masks[i])
            if ys.numel() == 0:
                boxes_xyxy[i] = torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device)
                areas[i] = 0.0
                continue
            y1 = ys.min().float()
            y2 = ys.max().float()
            x1 = xs.min().float()
            x2 = xs.max().float()
            boxes_xyxy[i, 0] = x1 / max(W - 1, 1)
            boxes_xyxy[i, 1] = y1 / max(H - 1, 1)
            boxes_xyxy[i, 2] = x2 / max(W - 1, 1)
            boxes_xyxy[i, 3] = y2 / max(H - 1, 1)
            areas[i] = ((x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)) / float(max(H * W, 1))
            soft_area = masks[i].mean()
            areas[i] = soft_area

        keep = (conf >= self.cfg.det_min_conf) & (areas >= self.cfg.det_min_area)
        idx = torch.nonzero(keep).squeeze(-1)

        if idx.numel() == 0:
            d = embs.shape[-1]
            return {
                "embs": torch.empty((0, d), device=self.device),
                "boxes": torch.empty((0, 4), device=self.device),
                "masks": torch.empty((0, H, W), device=self.device),
                "conf": torch.empty((0,), device=self.device),
                "cls": torch.empty((0,), device=self.device, dtype=torch.long),
                "depth": depth,
                "H": torch.tensor(H, device=self.device),
                "W": torch.tensor(W, device=self.device),
            }

        embs_k = embs[idx]
        boxes_k = boxes_xyxy[idx]
        masks_k = masks[idx]
        conf_k = conf[idx]
        cls_k = cls[idx]

        return {
            "embs": embs_k,
            "boxes": boxes_k,
            "masks": masks_k,
            "conf": conf_k,
            "cls": cls_k,
            "depth": depth,
            "H": torch.tensor(H, device=self.device),
            "W": torch.tensor(W, device=self.device),
        }

    def _ensure_erfg_entity(self, eid: int) -> None:
        if eid in self.erfg.entities:
            return

        pose_mean = torch.zeros(6, device=self.device)
        pose_cov = torch.eye(6, device=self.device) * 0.25
        vel_mean = torch.zeros(6, device=self.device)
        vel_cov = torch.eye(6, device=self.device) * 0.25

        node = EntityNode(
            entity_id=eid,
            entity_type=torch.zeros(32, device=self.device),
            pose=GaussianState(pose_mean, pose_cov),
            velocity=GaussianState(vel_mean, vel_cov),
            frame=self.erfg.frames["ego"],
        )
        self.erfg.add_entity(node)

    def _spawn_track(self, det_emb: torch.Tensor, det_box: torch.Tensor, det_depth: float) -> int:
        eid = self._new_entity_id()

        init = torch.tensor(
            [
                float((det_box[0] + det_box[2]) * 0.5),
                float((det_box[1] + det_box[3]) * 0.5),
                float(det_depth),
                0.0,
                0.0,
                0.0,
            ],
            device=self.device,
            dtype=torch.float32,
        )
        init_cov = torch.eye(6, device=self.device) * 0.15
        filt = CVFilter(init_mean=init, init_cov=init_cov, device=self.device)

        self._track_ids.append(eid)
        self._track_filters[eid] = filt
        self._track_embs[eid] = det_emb.detach()
        self._track_boxes[eid] = det_box.detach()
        self._track_missed[eid] = 0

        self._ensure_erfg_entity(eid)
        return eid

    def _track_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._track_ids:
            return (
                torch.empty((0, 256), device=self.device),
                torch.empty((0, 4), device=self.device),
            )
        embs = torch.stack([self._track_embs[eid] for eid in self._track_ids], dim=0)
        boxes = torch.stack([self._track_boxes[eid] for eid in self._track_ids], dim=0)
        return embs, boxes

    def _update_erfg_from_tracks(self) -> None:
        for eid in self._track_ids:
            self._ensure_erfg_entity(eid)
            node = self.erfg.entities[eid]

            filt = self._track_filters[eid]
            m = filt.state.mean
            P = filt.state.cov

            # Pose distribution
            node.pose.update(
                mean=torch.tensor([m[0], m[1], m[2], 0.0, 0.0, 0.0], device=self.device),
                cov=torch.diag(torch.tensor([P[0,0], P[1,1], P[2,2], 1.0, 1.0, 1.0], device=self.device)).to(self.device),
            )

            # Velocity distribution
            node.velocity.update(
                mean=torch.tensor([m[3], m[4], m[5], 0.0, 0.0, 0.0], device=self.device),
                cov=torch.diag(torch.tensor([P[3,3], P[4,4], P[5,5], 1.0, 1.0, 1.0], device=self.device)).to(self.device),
            )

    def _compute_relations_and_events(self, det: Dict) -> List[EventToken]:
        events: List[EventToken] = []
        eids = list(self._track_ids)
        if len(eids) < 2:
            self.erfg.relations = {}
            return events

        depth = det["depth"]
        H = int(det["H"].item())
        W = int(det["W"].item())

        relations_new = {}

        for i in range(len(eids)):
            for j in range(len(eids)):
                if i == j:
                    continue
                a = eids[i]
                b = eids[j]
                a_box = self._track_boxes[a]
                b_box = self._track_boxes[b]

                a_mask = torch.zeros((H, W), device=self.device)
                b_mask = torch.zeros((H, W), device=self.device)

                ax1 = int(max(0, min(W - 1, round(float(a_box[0]) * (W - 1)))))
                ay1 = int(max(0, min(H - 1, round(float(a_box[1]) * (H - 1)))))
                ax2 = int(max(0, min(W - 1, round(float(a_box[2]) * (W - 1)))))
                ay2 = int(max(0, min(H - 1, round(float(a_box[3]) * (H - 1)))))
                bx1 = int(max(0, min(W - 1, round(float(b_box[0]) * (W - 1)))))
                by1 = int(max(0, min(H - 1, round(float(b_box[1]) * (H - 1)))))
                bx2 = int(max(0, min(W - 1, round(float(b_box[2]) * (W - 1)))))
                by2 = int(max(0, min(H - 1, round(float(b_box[3]) * (H - 1)))))

                if ax2 > ax1 and ay2 > ay1:
                    a_mask[ay1:ay2, ax1:ax2] = 1.0
                if bx2 > bx1 and by2 > by1:
                    b_mask[by1:by2, bx1:bx2] = 1.0

                rel = infer_pair_relations(a_box, b_box, a_mask, b_mask, depth)

                edge = self.erfg.add_relation(a, b)
                for k, p in rel.items():
                    edge.set(k, float(p))
                    relations_new[(a, b, k)] = float(p)

                    prev = self._prev_rel_probs.get((a, b, k), 0.0)
                    th = self.cfg.relation_threshold
                    if prev < th <= p:
                        events.append(EventToken(t=self.erfg.time, kind=f"{k}_begin", src=a, dst=b, p=float(p)))
                    elif prev >= th > p:
                        events.append(EventToken(t=self.erfg.time, kind=f"{k}_end", src=a, dst=b, p=float(p)))

        self._prev_rel_probs = relations_new
        return events

    def step(self, evidence: Dict, dt: float) -> Dict:
        dt = float(max(dt, 1e-4))
        self.erfg.step_time(dt)

        det = self._extract_detections(evidence)
        det_embs = det["embs"]
        det_boxes = det["boxes"]
        depth_map = det["depth"]

        # Predict all tracks
        for eid in list(self._track_ids):
            self._track_filters[eid].predict(dt)

        # Associate
        trk_embs, trk_boxes = self._track_tensors()
        matches, unmatched_dets, unmatched_trks, _ = greedy_associate(
            track_embs=trk_embs,
            track_boxes=trk_boxes,
            det_embs=det_embs,
            det_boxes=det_boxes,
            min_score=self.cfg.assoc_min_score,
        )

        # Update matched
        for trk_idx, det_idx in matches.items():
            eid = self._track_ids[trk_idx]
            box = det_boxes[det_idx]
            emb = det_embs[det_idx]
            d = float(self._sample_depth(depth_map, box))

            z = torch.tensor(
                [
                    float((box[0] + box[2]) * 0.5),
                    float((box[1] + box[3]) * 0.5),
                    float(d),
                ],
                device=self.device,
            )

            # Measurement noise scales with instance uncertainty if available
            conf = float(det["conf"][det_idx].item()) if det["conf"].numel() else 0.5
            sigma = max(0.02, min(0.25, 0.25 * (1.0 - conf)))
            R = torch.eye(3, device=self.device) * (sigma ** 2)

            self._track_filters[eid].update(z, R)
            self._track_embs[eid] = emb.detach()
            self._track_boxes[eid] = box.detach()
            self._track_missed[eid] = 0

        # Handle unmatched tracks
        for trk_idx in unmatched_trks:
            eid = self._track_ids[trk_idx]
            self._track_missed[eid] += 1
            # Inflate covariance under occlusion
            f = self._track_filters[eid]
            f.state.cov = f.state.cov + torch.eye(6, device=self.device) * 0.01

        # Drop dead tracks
        alive = []
        for eid in self._track_ids:
            if self._track_missed[eid] <= self.cfg.max_missed:
                alive.append(eid)
            else:
                self.erfg.remove_entity(eid)
                self._track_filters.pop(eid, None)
                self._track_embs.pop(eid, None)
                self._track_boxes.pop(eid, None)
                self._track_missed.pop(eid, None)
        self._track_ids = alive

        # Spawn for unmatched detections
        for det_idx in unmatched_dets:
            if len(self._track_ids) >= self.cfg.max_entities:
                break
            box = det_boxes[det_idx]
            emb = det_embs[det_idx]
            d = float(self._sample_depth(depth_map, box))
            self._spawn_track(emb, box, d)

        # Write into ERFG
        self._update_erfg_from_tracks()

        # Relations and events
        events = self._compute_relations_and_events(det)

        return {
            "time": self.erfg.time,
            "erfg": self.erfg,
            "events": events,
        }

    @staticmethod
    def _sample_depth(depth: torch.Tensor, box_xyxy: torch.Tensor) -> float:
        H, W = int(depth.shape[-2]), int(depth.shape[-1])
        cx = float((box_xyxy[0] + box_xyxy[2]) * 0.5)
        cy = float((box_xyxy[1] + box_xyxy[3]) * 0.5)
        x = int(min(W - 1, max(0, round(cx * (W - 1)))))
        y = int(min(H - 1, max(0, round(cy * (H - 1)))))
        return float(depth[y, x])
