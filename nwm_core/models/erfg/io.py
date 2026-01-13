import torch
import json

def serialize_erfg(erfg):
    data = {
        "time": erfg.time,
        "entities": {},
        "relations": {}
    }

    for eid, e in erfg.entities.items():
        data["entities"][eid] = {
            "type": e.type_dist.tolist(),
            "pose_mean": e.pose.mean.tolist(),
            "pose_cov": e.pose.cov.tolist(),
            "vel_mean": e.velocity.mean.tolist(),
            "vel_cov": e.velocity.cov.tolist(),
            "affordances": e.affordances.tolist(),
        }

    for (a, b), r in erfg.relations.items():
        data["relations"][f"{a}->{b}"] = {
            "predicates": r.predicates,
            "parameters": r.parameters
        }

    return json.dumps(data)


def deserialize_erfg(blob):
    data = json.loads(blob)
    return data
