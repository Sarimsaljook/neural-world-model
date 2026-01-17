from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


@dataclass(frozen=True)
class Mask2FormerConfig:
    model_id: str = "facebook/mask2former-swin-large-coco-instance"
    max_queries: int = 100
    score_thresh: float = 0.35
    mask_thresh: float = 0.4
    input_short_side: int = 640
    input_max_size: int = 1024
    min_area_frac: float = 0.002


def _resize_keep_ar(x: torch.Tensor, short_side: int, max_size: int) -> Tuple[torch.Tensor, float]:
    b, c, h, w = x.shape
    scale = short_side / float(min(h, w))
    nh = int(round(h * scale))
    nw = int(round(w * scale))
    if max(nh, nw) > max_size:
        scale2 = max_size / float(max(nh, nw))
        scale *= scale2
        nh = int(round(h * scale))
        nw = int(round(w * scale))
    x2 = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
    return x2, float(scale)


def _boxes_xyxy_to_cxcywh_norm(boxes_xyxy: torch.Tensor, h: int, w: int) -> torch.Tensor:
    x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    bw = (x2 - x1).clamp(min=0.0)
    bh = (y2 - y1).clamp(min=0.0)
    out = torch.stack([cx / w, cy / h, bw / w, bh / h], dim=-1)
    return out.clamp(0.0, 1.0)


def _mask_to_box_xyxy(mask: torch.Tensor) -> Optional[torch.Tensor]:
    if mask.ndim != 2:
        return None
    ys, xs = torch.where(mask)
    if ys.numel() == 0:
        return None
    y1 = ys.min().float()
    y2 = ys.max().float()
    x1 = xs.min().float()
    x2 = xs.max().float()
    return torch.stack([x1, y1, x2, y2], dim=0)


def _post_to_instances_any(post_item: Any, H: int, W: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(post_item, dict) and ("scores" in post_item and "labels" in post_item and "masks" in post_item):
        scores = post_item["scores"].to(device).float()
        labels = post_item["labels"].to(device).long()
        masks = post_item["masks"].to(device)
        if masks.dtype != torch.float32:
            masks = masks.float()
        masks = masks.clamp(0.0, 1.0)
        return scores, labels, masks

    if isinstance(post_item, dict) and ("segmentation" in post_item and "segments_info" in post_item):
        seg = post_item["segmentation"].to(device)
        info = post_item["segments_info"]
        scores_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        masks_list: List[torch.Tensor] = []

        for s in info:
            sid = int(s.get("id", -1))
            if sid < 0:
                continue
            label_id = int(s.get("label_id", 0))
            score = float(s.get("score", 1.0))
            m = (seg == sid)
            area = float(m.float().sum().item()) / float(H * W)
            if area <= 0.0:
                continue
            scores_list.append(torch.tensor(score, device=device))
            labels_list.append(torch.tensor(label_id, device=device))
            masks_list.append(m.float())

        if len(scores_list) == 0:
            return (
                torch.zeros((0,), device=device),
                torch.zeros((0,), device=device, dtype=torch.long),
                torch.zeros((0, H, W), device=device),
            )

        scores = torch.stack(scores_list, dim=0)
        labels = torch.stack(labels_list, dim=0).long()
        masks = torch.stack(masks_list, dim=0).float().clamp(0.0, 1.0)
        return scores, labels, masks

    return (
        torch.zeros((0,), device=device),
        torch.zeros((0,), device=device, dtype=torch.long),
        torch.zeros((0, H, W), device=device),
    )


class Mask2FormerInstSeg(nn.Module):
    def __init__(self, cfg: Optional[Mask2FormerConfig] = None):
        super().__init__()
        self.cfg = cfg or Mask2FormerConfig()
        self.processor = AutoImageProcessor.from_pretrained(self.cfg.model_id)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(self.cfg.model_id)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, image_rgb_01: torch.Tensor, dino_feat_map: torch.Tensor, class_count: int) -> Dict[str, torch.Tensor]:
        device = image_rgb_01.device
        b, _, H, W = image_rgb_01.shape

        # Convert to uint8 HWC on CPU
        imgs_u8 = (image_rgb_01.clamp(0, 1) * 255.0).to(torch.uint8)
        imgs_u8 = imgs_u8.permute(0, 2, 3, 1).contiguous().cpu().numpy()  # (B,H,W,3) uint8

        inputs = self.processor(images=list(imgs_u8), return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        out = self.model(**inputs)

        target_sizes = [(H, W)] * b

        post = None
        try:
            post = self.processor.post_process_instance_segmentation(out, threshold=self.cfg.score_thresh, target_sizes=target_sizes)
        except Exception:
            post = None

        if post is None:
            try:
                post = self.processor.post_process_panoptic_segmentation(out, threshold=self.cfg.score_thresh, target_sizes=target_sizes)
            except Exception:
                post = None

        if post is None:
            post = [None] * b

        Q = int(self.cfg.max_queries)
        D = int(dino_feat_map.shape[1])

        mask_logits = torch.full((b, Q, H, W), -20.0, device=device, dtype=torch.float32)
        class_logits = torch.full((b, Q, class_count + 1), -10.0, device=device, dtype=torch.float32)
        boxes = torch.zeros((b, Q, 4), device=device, dtype=torch.float32)
        embs = torch.zeros((b, Q, D), device=device, dtype=torch.float32)
        valid = torch.zeros((b, Q), device=device, dtype=torch.bool)

        min_area = float(self.cfg.min_area_frac) * float(H * W)

        for i in range(b):
            scores, labels, masks = _post_to_instances_any(post[i], H, W, device)
            if scores.numel() == 0:
                continue

            areas = masks.flatten(1).sum(dim=1)
            keep = areas >= min_area
            if keep.sum().item() == 0:
                continue

            scores = scores[keep]
            labels = labels[keep]
            masks = masks[keep]

            sort_idx = torch.argsort(scores, descending=True)
            scores = scores[sort_idx]
            labels = labels[sort_idx]
            masks = masks[sort_idx]

            n = min(Q, int(scores.shape[0]))
            if n <= 0:
                continue

            scores_n = scores[:n].clamp(0.0, 1.0)
            labels_n = labels[:n].long().clamp(0, class_count - 1)
            masks_n = masks[:n].clamp(0.0, 1.0)

            mask_logits[i, :n] = torch.logit(masks_n.clamp(1e-4, 1 - 1e-4))

            class_logits[i, :n, :] = -10.0
            class_logits[i, :n, -1] = 0.0
            class_logits[i, torch.arange(n, device=device), labels_n] = torch.logit(scores_n.clamp(1e-4, 1 - 1e-4))

            for j in range(n):
                m = masks_n[j] > self.cfg.mask_thresh
                box = _mask_to_box_xyxy(m)
                if box is None:
                    continue
                boxes[i, j] = _boxes_xyxy_to_cxcywh_norm(box[None, :], H, W)[0]

            feat = dino_feat_map[i : i + 1]
            mask_small = F.interpolate(masks_n.unsqueeze(1), size=feat.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
            denom = mask_small.flatten(1).sum(dim=1).clamp(min=1e-6)
            pooled = (feat[0].unsqueeze(0) * mask_small.unsqueeze(1)).flatten(2).sum(dim=2) / denom.unsqueeze(1)
            embs[i, :n] = F.normalize(pooled, dim=-1)
            valid[i, :n] = True

        return {
            "mask_logits": mask_logits,
            "class_logits": class_logits,
            "boxes": boxes,
            "embeddings": embs,
            "valid": valid,
        }
