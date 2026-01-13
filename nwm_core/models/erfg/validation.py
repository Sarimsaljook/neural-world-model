def validate_erfg(erfg):
    for eid, ent in erfg.entities.items():
        assert ent.pose.mean.shape[-1] in (2, 3, 6)
        assert ent.pose.cov.shape[0] == ent.pose.cov.shape[1]

    for (a, b), rel in erfg.relations.items():
        assert a in erfg.entities
        assert b in erfg.entities
        for p, prob in rel.predicates.items():
            assert 0.0 <= prob <= 1.0

    return True
