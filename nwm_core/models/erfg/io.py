from __future__ import annotations

from typing import Any, Dict, List
import numpy as np

from .state import (
    Affordances, EntityNode, ERFGState, Frame, Gaussian, GeometryProxy,
    HypothesisComponent, PhysicalPropsBelief, RelationEdge, RelationParams,
    SE3Belief, VelocityBelief
)

def _arr(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)

def erfg_to_json(s: ERFGState) -> Dict[str, Any]:
    def g(g0: Gaussian) -> Dict[str, Any]:
        return {"mean": g0.mean.tolist(), "cov": g0.cov.reshape(-1).tolist()}

    def frame(f: Frame) -> Dict[str, Any]:
        return {"frame_id": f.frame_id, "rotation_mat": f.rotation_mat.reshape(-1).tolist(), "translation": f.translation.tolist()}

    def ent(e: EntityNode) -> Dict[str, Any]:
        return {
            "entity_id": e.entity_id,
            "type_logits": e.type_logits.tolist(),
            "pose": {"position": g(e.pose.position), "rotation": g(e.pose.rotation)},
            "velocity": {"linear": g(e.velocity.linear), "angular": g(e.velocity.angular)},
            "geometry": {"kind": e.geometry.kind, "params": {k: v.tolist() for k, v in e.geometry.params.items()}},
            "props": {
                "mass": g(e.props.mass), "friction": g(e.props.friction), "restitution": g(e.props.restitution),
                "stiffness": g(e.props.stiffness), "damping": g(e.props.damping)
            },
            "affordances": {
                "graspable": e.affordances.graspable, "pushable": e.affordances.pushable, "pullable": e.affordances.pullable,
                "openable": e.affordances.openable, "pourable": e.affordances.pourable, "fragile": e.affordances.fragile,
                "hot": e.affordances.hot, "sharp": e.affordances.sharp, "support_surface": e.affordances.support_surface,
                "container": e.affordances.container
            },
            "appearance_embed": e.appearance_embed.tolist(),
            "last_seen_ts": e.last_seen_ts,
            "alive_prob": e.alive_prob,
            "parts": {k: ent(v) for k, v in e.parts.items()},
            "extras": dict(e.extras or {})
        }

    def rel(r: RelationEdge) -> Dict[str, Any]:
        return {
            "src": r.src,
            "dst": r.dst,
            "predicate_logits": r.predicate_logits.tolist(),
            "predicate": r.predicate,
            "confidence": r.confidence,
            "params": {"params": {k: v.tolist() for k, v in r.params.params.items()}},
            "extras": dict(r.extras or {})
        }

    hyps = []
    for h in s.hypotheses:
        hyps.append({
            "weight": h.weight,
            "entities": {k: ent(v) for k, v in h.entities.items()},
            "relations": [rel(r) for r in h.relations],
            "world_frame": frame(h.world_frame),
            "extras": dict(h.extras or {})
        })

    return {
        "timestamp_ns": s.timestamp_ns,
        "version": s.version,
        "ego_frame": frame(s.ego_frame),
        "world_frame": frame(s.world_frame),
        "hypotheses": hyps,
        "active_entities": list(s.active_entities),
        "extras": dict(s.extras or {})
    }

def erfg_from_json(d: Dict[str, Any]) -> ERFGState:
    def g(x: Dict[str, Any], dim: int) -> Gaussian:
        mean = _arr(x["mean"]).reshape(dim)
        cov = _arr(x["cov"])
        if cov.size == dim * dim:
            cov = cov.reshape(dim, dim)
        else:
            cov = np.diag(cov.reshape(-1))
        return Gaussian(mean=mean, cov=cov)

    def frame(x: Dict[str, Any]) -> Frame:
        R = _arr(x["rotation_mat"]).reshape(3,3)
        t = _arr(x["translation"]).reshape(3)
        return Frame(frame_id=str(x["frame_id"]), rotation_mat=R, translation=t)

    def afford(a: Dict[str, Any]) -> Affordances:
        return Affordances(
            graspable=float(a["graspable"]), pushable=float(a["pushable"]), pullable=float(a["pullable"]),
            openable=float(a["openable"]), pourable=float(a["pourable"]), fragile=float(a["fragile"]),
            hot=float(a["hot"]), sharp=float(a["sharp"]), support_surface=float(a["support_surface"]), container=float(a["container"])
        )

    def ent(x: Dict[str, Any]) -> EntityNode:
        pose = SE3Belief(position=g(x["pose"]["position"], 3), rotation=g(x["pose"]["rotation"], 3))
        vel = VelocityBelief(linear=g(x["velocity"]["linear"], 3), angular=g(x["velocity"]["angular"], 3))
        props = PhysicalPropsBelief(
            mass=g(x["props"]["mass"], 1),
            friction=g(x["props"]["friction"], 1),
            restitution=g(x["props"]["restitution"], 1),
            stiffness=g(x["props"]["stiffness"], 1),
            damping=g(x["props"]["damping"], 1),
        )
        geom = GeometryProxy(kind=str(x["geometry"]["kind"]), params={k: _arr(v) for k, v in x["geometry"]["params"].items()})
        parts = {k: ent(v) for k, v in (x.get("parts") or {}).items()}
        return EntityNode(
            entity_id=str(x["entity_id"]),
            type_logits=_arr(x["type_logits"]),
            pose=pose,
            velocity=vel,
            geometry=geom,
            props=props,
            affordances=afford(x["affordances"]),
            appearance_embed=_arr(x["appearance_embed"]),
            last_seen_ts=int(x["last_seen_ts"]),
            alive_prob=float(x.get("alive_prob", 1.0)),
            parts=parts,
            extras=dict(x.get("extras") or {})
        )

    def rel(x: Dict[str, Any]) -> RelationEdge:
        params = RelationParams(params={k: _arr(v) for k, v in x["params"]["params"].items()})
        return RelationEdge(
            src=str(x["src"]),
            dst=str(x["dst"]),
            predicate_logits=_arr(x["predicate_logits"]),
            predicate=x.get("predicate"),
            confidence=float(x["confidence"]),
            params=params,
            extras=dict(x.get("extras") or {})
        )

    hyps = []
    for h in d["hypotheses"]:
        hyps.append(HypothesisComponent(
            weight=float(h["weight"]),
            entities={k: ent(v) for k, v in h["entities"].items()},
            relations=[rel(r) for r in h["relations"]],
            world_frame=frame(h["world_frame"]),
            extras=dict(h.get("extras") or {})
        ))

    return ERFGState(
        timestamp_ns=int(d["timestamp_ns"]),
        version=int(d.get("version", 0)),
        ego_frame=frame(d["ego_frame"]),
        world_frame=frame(d["world_frame"]),
        hypotheses=hyps,
        active_entities=list(d["active_entities"]),
        extras=dict(d.get("extras") or {})
    )
