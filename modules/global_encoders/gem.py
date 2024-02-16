import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class GeM(nn.Module):
    def __init__(self, feature_size=1024, output_dim=256, p=3, eps=1e-6, output_dim_matching=True):
        super().__init__()
        self.feature_size = feature_size
        self.output_dim = output_dim
        self.output_dim_matching = output_dim_matching
        self.eps = eps

        self.p = nn.Parameter(torch.ones(1) * p)
        if self.output_dim_matching:
            self.hidden_weights = nn.Parameter(1 / sqrt(self.feature_size) *
                                           torch.randn(self.feature_size, self.output_dim))       # output weights
            self.bn1 = nn.BatchNorm1d(self.output_dim)

    def forward(self, x):
        x_out = self.gem(x)
        x_out = x_out.squeeze(dim=-1).squeeze(dim=-1)

        # match output dimension
        if self.output_dim_matching:
            x_out = x_out @ self.hidden_weights
            x_out = self.bn1(x_out)

        return x_out

    def gem(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)

        # # use 3d average pooling layer to match the output dimension
        # assert self.feature_size % self.output_dim == 0
        # return F.avg_pool3d(x.clamp(min=self.eps).pow(self.p),
        #                     (self.feature_size // self.output_dim, x.size(-2), x.size(-1)))


if __name__ == '__main__':
    gem = GeM(feature_size=1024, output_dim=256, p=3, eps=1e-6, output_dim_matching=True)

    torch.manual_seed(123)
    input1 = F.normalize(torch.randn((3, 1024, 900, 1)), dim=1)
    input2 = input1.clone()
    input2 = input2[:, :, torch.randperm(input2.shape[2]), :]
    diff = torch.norm(input2 - input1)

    print("==================================")
    print(f'Norm of the difference before GeM: {diff:.3f}.')

    with torch.no_grad():
        gem.eval()
        output1 = gem(input1)
        output2 = gem(input2)
        diff = torch.norm(output2 - output1)
        print(f'Norm of the difference after GeM: {diff:.3f}.')
        print(output1.shape)
