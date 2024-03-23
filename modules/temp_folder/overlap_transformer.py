import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.global_encoders.netvlad import NetVLADLoupe
from modules.local_encoders.overlap_net_leg_32 import OverlapNetLeg32


class featureExtracter(nn.Module):
    def __init__(self, height=32, width=900, channels=1, norm_layer=None, use_transformer=True):
        super(featureExtracter, self).__init__()
        self.overlap_net_leg32 = OverlapNetLeg32(height, width, channels, norm_layer, use_transformer)
        self.net_vlad = NetVLADLoupe(feature_size=1024, max_samples=900, cluster_size=64,
                                     output_dim=256, gating=True, add_batch_norm=False)

    def forward(self, x):
        x = self.overlap_net_leg32(x)
        x = self.net_vlad(x)
        return x
