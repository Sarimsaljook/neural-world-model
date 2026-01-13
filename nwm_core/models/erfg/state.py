
class GaussianState:
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov

    def update(self, mean, cov):
        self.mean = mean
        self.cov = cov


class PhysicalProperties:
    def __init__(self, mass, friction, stiffness):
        self.mass = mass
        self.friction = friction
        self.stiffness = stiffness
