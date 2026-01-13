import torch
from .state import GaussianState, PhysicalProperties
from .frames import Frame

class EntityNode:
    def __init__(self, entity_id, entity_type, pose, velocity, frame):
        self.id = entity_id
        self.type_dist = entity_type
        self.pose = pose
        self.velocity = velocity
        self.frame = frame
        self.geometry = None
        self.affordances = torch.zeros(16)
        self.physical = PhysicalProperties(
            mass=GaussianState(torch.tensor([1.0]), torch.tensor([[0.5]])),
            friction=GaussianState(torch.tensor([0.5]), torch.tensor([[0.2]])),
            stiffness=GaussianState(torch.tensor([1.0]), torch.tensor([[0.3]])),
        )
        self.history = []


class RelationEdge:
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
        self.predicates = {}
        self.parameters = {}

    def set(self, name, prob, params=None):
        self.predicates[name] = prob
        if params is not None:
            self.parameters[name] = params


class ERFG:
    def __init__(self):
        self.entities = {}
        self.relations = {}
        self.frames = {
            "world": Frame("world"),
            "ego": Frame("ego"),
        }
        self.time = 0.0

    def add_entity(self, node):
        self.entities[node.id] = node

    def remove_entity(self, entity_id):
        self.entities.pop(entity_id, None)
        self.relations = {
            k: v for k, v in self.relations.items()
            if entity_id not in k
        }

    def add_relation(self, src, dst):
        key = (src, dst)
        if key not in self.relations:
            self.relations[key] = RelationEdge(src, dst)
        return self.relations[key]

    def step_time(self, dt):
        self.time += dt
