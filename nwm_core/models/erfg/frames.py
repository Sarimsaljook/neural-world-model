import torch

class Frame:
    def __init__(self, name, parent=None, T_parent=None):
        self.name = name
        self.parent = parent
        self.T_parent = T_parent if T_parent is not None else torch.eye(4)

    def world_T(self):
        if self.parent is None:
            return self.T_parent
        return self.parent.world_T() @ self.T_parent

    def transform(self, T_new):
        self.T_parent = T_new
