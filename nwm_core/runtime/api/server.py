from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ...common.logging import get_logger
from ...models.erfg.io import erfg_to_json
from ..loop.realtime import RealtimeEngine, RealtimeConfig
from ..ingest.camera import CameraConfig
from ..loop.scheduler import RuntimeClocks

log = get_logger("nwm.api")


class PredictRequest(BaseModel):
    horizon: int = 15  # frames
    dt: float = 1.0


class CounterfactualRequest(BaseModel):
    horizon: int = 15
    dt: float = 1.0
    interventions: Dict[str, Any] = {}  # e.g. {"remove": ["id1"], "velocity_delta": {"id2":[10,0]}}


def create_app(engine: RealtimeEngine) -> FastAPI:
    app = FastAPI(title="Neural World Model API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"ok": True}

    @app.get("/state")
    def state() -> Dict[str, Any]:
        return engine.export_state()

    @app.post("/predict")
    def predict(req: PredictRequest) -> Dict[str, Any]:
        return engine.predict(horizon=req.horizon, dt=req.dt)

    @app.post("/counterfactual")
    def counterfactual(req: CounterfactualRequest) -> Dict[str, Any]:
        return engine.counterfactual(horizon=req.horizon, dt=req.dt, interventions=req.interventions)

    @app.get("/probes")
    def probes() -> Dict[str, Any]:
        return {"probes": [p.__dict__ for p in engine.propose_probes()]}

    @app.post("/memory/consolidate")
    def consolidate() -> Dict[str, Any]:
        engine.consolidate_memory()
        return {"ok": True}

    return app


def build_default_engine() -> RealtimeEngine:
    cfg = RealtimeConfig(camera=CameraConfig(), clocks=RuntimeClocks())
    return RealtimeEngine(cfg)
