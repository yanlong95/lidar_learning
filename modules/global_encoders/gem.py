import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    """
    The GeM pooling layer.

    Args:
        feature_size: (int) The size of the feature layers (channels).
        output_size: (int) The dimension of the output vector.
        p: (float) Norm initialization parameter (default: 3 based on the paper of GeM)
        eps: (float) The threshold of clamping to keep the values of x be positive.
        share_p: (bool) Whether to use shared parameter p between all feature layers.
    """
    def __init__(self, feature_size=1024, output_size=256, p=3, eps=1e-6, share_p=True):
        super().__init__()
        self.feature_size = feature_size
        self.output_size = output_size
        self.eps = eps
        self.share_p = share_p

        # share p for all feature layers or each feature layer has a p
        if self.share_p:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.p = nn.Parameter(torch.ones(feature_size) * p)

        # match the output vector size
        self.avg_pool = nn.AvgPool2d((feature_size // output_size, 1))

    def forward(self, x):
        """
        feedforward pass of the NetVLAD neural network

        Args:
            x: (torch.tensor) with dimension (batch_size, feature_size, local_descriptor_size, 1)
                in form [B, C, W, H].
        """
        x_out = self.gem(x)
        x_out = x_out.squeeze(-1)
        # x_out = self.avg_pool(x_out)    # match the output dimension by average pooling
        x_out = x_out.squeeze(-1)
        return x_out

    def gem(self, x):
        """
        gem function to compute the generalize mean of each feature layers.
        If share p is True, all feature layers share the same parameter p.

        Args:
            x: (torch.tensor) with dimension (batch_size, feature_size, local_descriptor_size, 1)
                in form [B, C, W, H].
        """
        if self.share_p:
            return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)
        else:
            x_out = x.transpose(1, 3)                                           # .pow apply at axis -1
            x_out = x_out.clamp(min=self.eps).pow(self.p)
            x_out = x_out.transpose(1, 3)                                       # avg_pool apply at the last two axis
            x_out = F.avg_pool2d(x_out, (x.size(-2), x.size(-1)))
            x_out = x_out.transpose(1, 3)                            # .pow apply at axis -1
            x_out = x_out.pow(1./self.p)
            x_out = x_out.transpose(1, 3)
            return x_out


if __name__ == '__main__':
    gem = GeM(feature_size=1024, output_size=256, p=3, eps=1e-6, share_p=True)

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
