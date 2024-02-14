import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class NetVLADLoupe(nn.Module):
    """
    The netvald neural network, details check the paper NetVLAD and paper PointNetVLAD.
    """
    def __init__(self, feature_size, max_samples, cluster_size, output_dim,
                 gating=True, add_batch_norm=True, is_training=True):
        super().__init__()
        self.feature_size = feature_size
        self.max_samples = max_samples
        self.cluster_size = cluster_size
        self.output_dim = output_dim
        self.gating = gating
        self.add_batch_norm = add_batch_norm
        self.is_training = is_training
        self.softmax = nn.Softmax(dim=-1)

        # add weights variables
        self.cluster_weights = nn.Parameter(1 / sqrt(feature_size) * torch.randn(feature_size, cluster_size))
        self.cluster_weights2 = nn.Parameter(1 / sqrt(feature_size) * torch.randn(1, feature_size, cluster_size))
        self.hidden_weights = nn.Parameter(1 / sqrt(feature_size) * torch.randn(feature_size * cluster_size, output_dim))

        # determine cluster norm and biases
        if add_batch_norm:
            self.bn1 = nn.BatchNorm1d(cluster_size)
            self.cluster_biases = None
        else:
            self.bn1 = None
            self.cluster_biases = nn.Parameter(1 / sqrt(feature_size) * torch.randn(cluster_size))

        self.bn2 = nn.BatchNorm1d(output_dim)

        # apply GatingContext filter before output the descriptor
        if self.gating:
            self.context_gating = GatingContext(output_dim, add_batch_norm=add_batch_norm)

    def forward(self, x):
        """
        feedforward pass of the netvald neural network

        Args:
            x: (torch.tensor) with dimension (batch_size, feature_size, max_samples, 1).
                Considering the output of the last layer of the previous nn as H * W * D, which can be considered as
                D dimensional descriptors extracted at H * W spatial locations.
                Here H = 1, W = max_samples, D = feature_size.
        """
        pass


class GatingContext(nn.Module):
    """
    A filter applies before the output of netvlad descriptor.
    """

    def __init__(self, dim, add_batch_norm=True):
        """
        Define a small filter to determine if the weight of each element in the netvald descriptor.

        Args:
            dim: (int) the dimension of the descriptor output by netvald.
            add_batch_norm: (bool) whether to use batch normalization (if not add biases).
        """
        super().__init__()
        self.dim = dim
        self.add_batch_norm = add_batch_norm
        self.gating_weights = nn.Parameter(1 / sqrt(dim) * torch.randn(dim, dim))
        self.sigmoid = nn.Sigmoid()

        if self.add_batch_norm:
            self.gating_biases = None
            self.bn1 = nn.BatchNorm1d(dim)
        else:
            self.gating_biases = nn.Parameter(1 / sqrt(dim) * torch.randn(dim))
            self.bn1 = None

    def forward(self, x):
        gates = x @ self.gating_weights

        if self.add_batch_norm:
            gates = self.bn1(gates)
        else:
            gates = gates + self.gating_biases

        gates = self.sigmoid(gates)
        activation = x * gates

        return activation


if __name__ == '__main__':
    # net_vlad = NetVLADLoupe(feature_size=1024, max_samples=900, cluster_size=64,
    #                         output_dim=256, gating=True, add_batch_norm=False,
    #                         is_training=True)

    net_vlad = NetVLADLoupe(feature_size=1024, max_samples=900, cluster_size=16,
                            output_dim=20, gating=True, add_batch_norm=True,
                            is_training=True)
    # input  (bs, 1024, 360, 1)
    torch.manual_seed(1234)
    input_tensor = F.normalize(torch.randn((2, 1024, 900, 1)), dim=1)
    print("==================================")

    with torch.no_grad():
        net_vlad.eval()
        print(input_tensor.shape)
        # out1 = net_vlad(input_tensor)
        # print(out1.shape)
        # net_vlad.eval()
        # # input_tensor2[:, :, 20:, :] = 0.1
        # input_tensor2 = F.normalize(input_tensor2, dim=1)
        # out2 = net_vlad(input_tensor2)
        # print(out2)
        # net_vlad.eval()
        # input_tensor3 = torch.randn((1, 1024, 360, 1))
        # out3 = net_vlad(input_tensor3)
        # print(out3)
        #
        # print(((out1 - out2) ** 2).sum(1))
        # print(((out1 - out3) ** 2).sum(1))
