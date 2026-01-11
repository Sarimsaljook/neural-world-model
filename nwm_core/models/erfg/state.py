from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal
import numpy as np

Predicate = Literal[
    "contact", "supporting", "supported_by", "inside", "contains",
    "attached", "hinge", "slider", "held_by", "occludes", "left_of",
    "right_of", "in_front_of", "behind"
]

EventType = Literal[
    "contact_begin", "contact_end", "grasp_begin", "grasp_end",
    "attach_inferred", "detach_inferred", "spill", "tip", "jam",
    "slide", "hinge_rotate", "collision", "occlusion_enter", "occlusion_exit",
    "support_lost", "support_gained"
]

@dataclass
class Gaussian:
    mean: np.ndarray
    cov: np.ndarray

@dataclass
class SE3Belief:
    position: Gaussian
    rotation: Gaussian  # axis-angle

@dataclass
class VelocityBelief:
    linear: Gaussian
    angular: Gaussian

@dataclass
class PhysicalPropsBelief:
    mass: Gaussian
    friction: Gaussian
    restitution: Gaussian
    stiffness: Gaussian
    damping: Gaussian

@dataclass
class GeometryProxy:
    kind: Literal["bbox", "sdf_grid", "capsule", "mesh_coarse"]
    params: Dict[str, np.ndarray]

@dataclass
class Affordances:
    graspable: float
    pushable: float
    pullable: float
    openable: float
    pourable: float
    fragile: float
    hot: float
    sharp: float
    support_surface: float
    container: float

@dataclass
class EntityNode:
    entity_id: str
    type_logits: np.ndarray
    pose: SE3Belief
    velocity: VelocityBelief
    geometry: GeometryProxy
    props: PhysicalPropsBelief
    affordances: Affordances
    appearance_embed: np.ndarray
    last_seen_ts: int
    alive_prob: float = 1.0
    parts: Dict[str, "EntityNode"] = field(default_factory=dict)
    extras: Dict = field(default_factory=dict)

@dataclass
class RelationParams:
    params: Dict[str, np.ndarray]

@dataclass
class RelationEdge:
    src: str
    dst: str
    predicate_logits: np.ndarray
    predicate: Optional[Predicate]
    confidence: float
    params: RelationParams
    extras: Dict = field(default_factory=dict)

@dataclass
class Frame:
    frame_id: str
    rotation_mat: np.ndarray  # (3,3)
    translation: np.ndarray   # (3,)

@dataclass
class HypothesisComponent:
    weight: float
    entities: Dict[str, EntityNode]
    relations: List[RelationEdge]
    world_frame: Frame
    extras: Dict = field(default_factory=dict)

@dataclass
class ERFGState:
    timestamp_ns: int
    ego_frame: Frame
    world_frame: Frame
    hypotheses: List[HypothesisComponent]
    active_entities: List[str]
    version: int = 0
    extras: Dict = field(default_factory=dict)

@dataclass
class EventToken:
    event_type: EventType
    timestamp_ns: int
    participants: List[str]
    confidence: float
    params: Dict
    extras: Dict = field(default_factory=dict)
