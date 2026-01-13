import torch.nn as nn
import torch.nn.functional as F

class DepthHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim // 2, 1, 1),
        )

    def forward(self, feats):
        depth = self.decoder(feats)
        depth = F.interpolate(depth, scale_factor=14, mode="bilinear", align_corners=False)
        return depth.squeeze(1)
