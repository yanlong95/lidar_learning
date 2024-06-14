import torch
import torch.nn as nn

from modules.local_encoders.overlap_net_leg_32 import OverlapNetLeg32
from modules.global_encoders.netvlad import NetVLADLoupe


class OverlapTransformer32(nn.Module):
    def __init__(self, height=32, width=512, channels=1, norm_layer=None, use_transformer=True):
        super(OverlapTransformer32, self).__init__()
        self.overlap_net_leg32 = OverlapNetLeg32(height, width, channels, norm_layer, use_transformer)
        self.net_vlad = NetVLADLoupe(feature_size=1024, max_samples=width, cluster_size=64,
                                     output_dim=256, gating=True, add_batch_norm=False)

    def forward(self, x):
        x = self.overlap_net_leg32(x)
        x = self.net_vlad(x)
        return x
