import torch.nn as nn

class KeypointHead(nn.Module):
    def __init__(self, dim, num_kp=21):
        super().__init__()
        self.head = nn.Linear(dim, num_kp * 2)

    def forward(self, inst_feats):
        return self.head(inst_feats).view(inst_feats.size(0), inst_feats.size(1), -1, 2)
