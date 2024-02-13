import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class netvald(nn.Module):
    pass


class GatingContext(nn.Module):
    def __init__(self, dim, add_batch_norm=True):
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
