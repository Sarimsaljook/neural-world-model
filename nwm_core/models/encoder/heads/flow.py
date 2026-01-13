import torch
import torch.nn as nn

class FlowHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, 2, 1),
        )

    def forward(self, feats_t, feats_t1):
        x = torch.cat([feats_t, feats_t1], dim=1)
        flow = self.net(x)
        return flow
