from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from ...common.types import BBox


def _as_bbox(e: Dict[str, Any]) -> BBox:
    xyxy = e.get("extras", {}).get("bbox_xyxy")
    if xyxy is None:
        xyxy = e.get("geometry", {}).get("params", {}).get("xyxy")
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return BBox(x1, y1, x2, y2)


def draw_overlay(
    frame_rgb: np.ndarray,
    state_json: Dict[str, Any],
    max_events: int = 6,
) -> np.ndarray:
    img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    hyp = (state_json.get("hypotheses") or [{}])[0]
    entities = hyp.get("entities", {})
    relations = hyp.get("relations", [])
    intuition = state_json.get("intuition", {})
    uncertainty = state_json.get("uncertainty", {})
    events = state_json.get("events", [])[-max_events:]

    # draw boxes
    for eid, ent in entities.items():
        b = _as_bbox(ent)
        x1, y1, x2, y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        u = float(uncertainty.get(eid, 0.0))
        stab = float(intuition.get(eid, {}).get("stability_risk", 0.0))
        slip = float(intuition.get(eid, {}).get("slip_risk", 0.0))

        label = f"{eid}  u={u:.1f}  stab={stab:.2f}  slip={slip:.2f}"
        cv2.putText(img, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # relations list
    y = 18
    cv2.putText(img, "Relations:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
    y += 18
    for r in relations[:10]:
        src = r.get("src")
        dst = r.get("dst")
        typ = r.get("predicate") or r.get("type")
        conf = float(r.get("confidence", 0.0))
        cv2.putText(img, f"{src}->{dst}: {typ} ({conf:.2f})", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        y += 16

    # events list
    y += 8
    cv2.putText(img, "Events:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 1, cv2.LINE_AA)
    y += 18
    for e in events:
        cv2.putText(img, f'{e.get("event_type")} {e.get("participants")}', (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        y += 16

    world_u = float(uncertainty.get("_world", 0.0))
    cv2.putText(img, f"World uncertainty: {world_u:.2f}", (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return img
