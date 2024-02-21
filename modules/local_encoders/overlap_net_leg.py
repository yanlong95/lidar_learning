import torch
import torch.nn as nn
import torch.nn.functional as F


class OverlapNetLeg(nn.Module):
    def __init__(self, height=64, width=900, channels=1, norm_layer=None, use_transformer=True):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.use_transformer = use_transformer

        # define layers
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=(5, 1), stride=(1, 1), bias=False)     # 896
        self.bn1 = norm_layer(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 1), stride=(2, 1), bias=False)           # 447
        self.bn2 = norm_layer(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 1), stride=(2, 1), bias=False)           # 222
        self.bn3 = norm_layer(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 1), stride=(2, 1), bias=False)           # 110
        self.bn4 = norm_layer(64)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(2, 1), stride=(2, 1), bias=False)          # 55
        self.bn5 = norm_layer(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(2, 1), bias=False)         # 28
        self.bn6 = norm_layer(128)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(2, 1), bias=False)         # 14
        self.bn7 = norm_layer(128)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(2, 1), bias=False)         # 7
        self.bn8 = norm_layer(128)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(2, 1), bias=False)         # 4
        self.bn9 = norm_layer(128)
        self.conv10 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(2, 1), bias=False)        # 2
        self.bn10 = norm_layer(128)
        self.conv11 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(2, 1), bias=False)        # 1
        self.bn11 = norm_layer(128)
        self.relu = nn.ReLU()

        # add MHSA
        # maybe add more heads or layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=1024, activation='relu',
                                                   dropout=0.0)
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

        # batch, features, height, weight
        x = x.permute(0, 1, 3, 2)
        x = self.relu(self.convLast1(x))

        if self.use_transformer:
            x = x.squeeze(-1)
            # transformer batch_first = False, (seq, batch, feature)
            x = x.permute(2, 0, 1)
            x = self.transformer_encoder(x)
            x = x.permute(1, 2, 0)
            x = x.unsqueeze(-1)
            x = torch.cat((x, x), dim=1)
            x = self.relu(self.convLast2(x))




        