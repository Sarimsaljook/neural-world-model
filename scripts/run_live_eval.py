from __future__ import annotations

import cv2

from nwm_core.runtime.api.server import build_default_engine
from nwm_core.runtime.viz.overlay import draw_overlay


def main() -> None:
    engine = build_default_engine()

    while True:
        frame = engine.step()
        if frame is None:
            continue

        state = engine.export_state()
        vis = draw_overlay(frame, state)

        cv2.imshow("NWM Live (ERFG + Relations + Events + Intuition)", vis)
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord("q"):
            break

    engine.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
