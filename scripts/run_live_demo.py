from __future__ import annotations

import uvicorn

from nwm_core.runtime.api.server import build_default_engine, create_app


def main() -> None:
    engine = build_default_engine()
    app = create_app(engine)

    import threading
    import time

    def loop():
        while True:
            engine.step()
            time.sleep(0.001)

    t = threading.Thread(target=loop, daemon=True)
    t.start()

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
