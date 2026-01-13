from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

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
    suppress_iou: float = 0.85
    topk_keep: int = 40


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
    return x2, scale


def _boxes_xyxy_to_cxcywh_norm(boxes_xyxy: torch.Tensor, h: int, w: int) -> torch.Tensor:
    x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    bw = (x2 - x1).clamp(min=0.0)
    bh = (y2 - y1).clamp(min=0.0)
    out = torch.stack([cx / w, cy / h, bw / w, bh / h], dim=-1)
    return out.clamp(0.0, 1.0)


def _mask_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    inter = (a & b).sum().float()
    union = (a | b).sum().float().clamp(min=1.0)
    return inter / union


def _suppress_topk(
    masks: torch.Tensor, labels: torch.Tensor, scores: torch.Tensor, cfg: Mask2FormerConfig
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if scores.numel() == 0:
        return masks, labels, scores

    order = torch.argsort(scores, descending=True)
    masks = masks[order]
    labels = labels[order]
    scores = scores[order]

    kept: List[int] = []
    binm = masks > cfg.mask_thresh

    for i in range(binm.shape[0]):
        if scores[i] < cfg.score_thresh:
            break
        ok = True
        for j in kept:
            if labels[i].item() != labels[j].item():
                continue
            if float(_mask_iou(binm[i], binm[j]).item()) >= cfg.suppress_iou:
                ok = False
                break
        if ok:
            kept.append(i)
        if len(kept) >= cfg.topk_keep:
            break

    if not kept:
        return masks[:0], labels[:0], scores[:0]

    idx = torch.tensor(kept, device=masks.device, dtype=torch.long)
    return masks[idx], labels[idx], scores[idx]


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
    def forward(
        self,
        image_rgb_01: torch.Tensor,   # (B,3,H,W) float in [0,1]
        dino_feat_map: torch.Tensor,  # (B,D,Hp,Wp)
        class_count: int,
    ) -> Dict[str, torch.Tensor]:
        device = image_rgb_01.device
        b, _, H, W = image_rgb_01.shape

        # Convert to uint8 HWC on CPU
        imgs_u8 = (image_rgb_01.clamp(0, 1) * 255.0).to(torch.uint8)
        imgs_u8 = imgs_u8.permute(0, 2, 3, 1).contiguous().cpu().numpy()  # (B,H,W,3) uint8

        inputs = self.processor(images=list(imgs_u8), return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        out = self.model(**inputs)

        inst_list = self.processor.post_process_instance_segmentation(
            out, threshold=0.0, target_sizes=[(H, W)] * b
        )

        print("[Mask2Former] keys:", None if inst_list[0] is None else list(inst_list[0].keys()))

        Q = int(self.cfg.max_queries)
        D = int(dino_feat_map.shape[1])

        mask_logits = torch.full((b, Q, H, W), -20.0, device=device, dtype=torch.float32)
        class_logits = torch.full((b, Q, class_count + 1), -10.0, device=device, dtype=torch.float32)
        boxes = torch.zeros((b, Q, 4), device=device, dtype=torch.float32)
        embs = torch.zeros((b, Q, D), device=device, dtype=torch.float32)

        ys, xs = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij",
        )

        for i in range(b):
            item = inst_list[i]
            if item is None or not isinstance(item, dict):
                continue

            masks_i: Optional[torch.Tensor] = None
            labels_i: Optional[torch.Tensor] = None
            scores_i: Optional[torch.Tensor] = None

            if "masks" in item and "labels" in item:
                masks_i = item["masks"].to(device).float()
                labels_i = item["labels"].to(device).long()
                if "scores" in item:
                    scores_i = item["scores"].to(device).float()
                else:
                    scores_i = torch.ones((labels_i.numel(),), device=device, dtype=torch.float32)

            elif "segmentation" in item and "segments_info" in item:
                seg = item["segmentation"].to(device)
                seginfo = item["segments_info"]
                masks_acc: List[torch.Tensor] = []
                labels_acc: List[int] = []
                scores_acc: List[float] = []
                for s in seginfo:
                    sid = int(s.get("id", -1))
                    if sid < 0:
                        continue
                    cat = int(s.get("label_id", s.get("category_id", -1)))
                    if cat < 0:
                        continue
                    score = float(s.get("score", 1.0))
                    m = (seg == sid)
                    if torch.any(m):
                        masks_acc.append(m.float())
                        labels_acc.append(cat)
                        scores_acc.append(score)
                if masks_acc:
                    masks_i = torch.stack(masks_acc, dim=0)
                    labels_i = torch.tensor(labels_acc, device=device, dtype=torch.long)
                    scores_i = torch.tensor(scores_acc, device=device, dtype=torch.float32)

            if masks_i is None or labels_i is None or scores_i is None:
                continue

            masks_i, labels_i, scores_i = _suppress_topk(masks_i, labels_i, scores_i, self.cfg)
            n = int(min(Q, masks_i.shape[0]))
            if n == 0:
                continue

            masks_i = masks_i[:n].clamp(0.0, 1.0)
            labels_i = labels_i[:n].clamp(0, class_count - 1)
            scores_i = scores_i[:n].clamp(0.0, 1.0)

            mask_logits[i, :n] = torch.logit(masks_i.clamp(1e-4, 1 - 1e-4))

            class_logits[i, :n, :] = -10.0
            class_logits[i, :n, -1] = 0.0
            for j in range(n):
                cls = int(labels_i[j].item())
                sc = float(scores_i[j].item())
                class_logits[i, j, cls] = torch.logit(torch.tensor(sc, device=device).clamp(1e-4, 1 - 1e-4))

            for j in range(n):
                m = masks_i[j] > self.cfg.mask_thresh
                if not torch.any(m):
                    continue
                xj = xs[m]
                yj = ys[m]
                x1 = xj.min().float()
                x2 = xj.max().float()
                y1 = yj.min().float()
                y2 = yj.max().float()
                boxes_xyxy = torch.tensor([x1, y1, x2, y2], device=device)
                boxes[i, j] = _boxes_xyxy_to_cxcywh_norm(boxes_xyxy[None, :], H, W)[0]

            feat = dino_feat_map[i : i + 1]  # (1,D,Hp,Wp)
            mask_small = F.interpolate(
                masks_i[:n].unsqueeze(1),
                size=feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)  # (n,Hp,Wp)

            denom = mask_small.flatten(1).sum(dim=1).clamp(min=1e-6)
            pooled = (feat[0].unsqueeze(0) * mask_small.unsqueeze(1)).flatten(2).sum(dim=2) / denom.unsqueeze(1)
            embs[i, :n] = F.normalize(pooled, dim=-1)

        return {
            "mask_logits": mask_logits,
            "class_logits": class_logits,
            "boxes": boxes,
            "embeddings": embs,
        }
