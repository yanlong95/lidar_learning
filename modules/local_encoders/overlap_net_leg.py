"""
Code taken from https://github.com/haomo-ai/OverlapTransformer/tree/master/modules with trivial modifications.
    - drop the NetVLAD layer in the original code.
    - use batch_first-True in transformer.
    - the original code in condition transformer=False has an output in shape (1, 512, width, 1) which does not match
      the shape requirement in paper (1, 1024, width, 1). Thus, add an addition layer to double the feature layer size.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OverlapNetLeg(nn.Module):
    """
    The NetVLAD neural network, details check the paper NetVLAD and paper PointNetVLAD.

    Args:
        height: (int) The height of the image.
        width: (int) The width of the image.
        channels: (int) The number of channels of the image.
        norm_layer: (nn.Module) The normalization layer (default: nn.BatchNorm2d).
        use_transformer: (bool) Whether to use transformer.
    """
    def __init__(self, height=64, width=900, channels=1, norm_layer=None, use_transformer=True):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.use_transformer = use_transformer

        # define layers input = (batch, feature, height, width), output = (batch, feature, 1, width)
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=(5, 1), stride=(1, 1), bias=False)               # 30
        self.bn1 = norm_layer(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 1), stride=(2, 1), bias=False)           # 14
        self.bn2 = norm_layer(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 1), stride=(2, 1), bias=False)           # 6
        self.bn3 = norm_layer(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 1), stride=(2, 1), bias=False)           # 2
        self.bn4 = norm_layer(64)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(2, 1), stride=(2, 1), bias=False)          # 1
        self.bn5 = norm_layer(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(2, 1), bias=False)         # 1
        self.bn6 = norm_layer(128)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(2, 1), bias=False)         # 1
        self.bn7 = norm_layer(128)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(2, 1), bias=False)         # 1
        self.bn8 = norm_layer(128)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(2, 1), bias=False)         # 1
        self.bn9 = norm_layer(128)
        self.conv10 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(2, 1), bias=False)        # 1
        self.bn10 = norm_layer(128)
        self.conv11 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(2, 1), bias=False)        # 1
        self.bn11 = norm_layer(128)
        self.relu = nn.ReLU()

        # add MHSA
        # maybe add more heads or layers (original code use batch_first=False)
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=1024, activation='relu',
                                                   dropout=0.0, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)   # num_layers = 1 for efficient

        self.convLast1 = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bnLast1 = norm_layer(256)
        self.convLast2 = nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bnLast2 = norm_layer(512)

        # linear mapping
        self.linear = nn.Linear(128 * width, 256)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

        # attach NetVLAD or GeM later (base on OverlapTransformer add_batch_norm should be False in NetVLAD).

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = self.relu(self.conv9(x))
        x = self.relu(self.conv10(x))
        x = self.relu(self.conv11(x))

        x = x.permute(0, 1, 3, 2)                       # to (batch, features, width, 1)
        x = self.relu(self.convLast1(x))

        if self.use_transformer:
            # transformer branch
            out = x.squeeze(-1)                         # to (batch, feature, seq)
            out = out.permute(0, 2, 1)                  # transformer batch_first = True, input = (batch, seq, feature)
            out = self.transformer_encoder(out)
            out = out.permute(0, 2, 1)                  # to (batch, feature, seq), batch_first
            out = out.unsqueeze(-1)                     # to (batch, feature, width, 1)
            # concatenate original branch and transformer branch
            x = torch.cat((x, out), dim=1)       # to (batch, 2 * feature, width, 1)
            x = self.relu(self.convLast2(x))            # feature: 512 -> 1024
        else:
            x = torch.cat((x, x), dim=1)
            x = self.relu(self.convLast2(x))            # original code do not contain this line, maybe miss by mistake
        x = F.normalize(x, dim=1)
        # connect to NetVLAD or GeM and then normalize along feature axis (dim=1)

        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = OverlapNetLeg(64, 900, 1, use_transformer=True).to(device).eval()

    input1 = torch.randn((1, 1, 64, 900)).to(device)
    output1 = encoder(input1)

    print(output1.shape)
