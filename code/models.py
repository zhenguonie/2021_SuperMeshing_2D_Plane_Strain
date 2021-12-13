import torch
from torch import nn
import torch.nn.functional as F
import math


class BN_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, bias=True):
        super(BN_Conv2d, self).__init__()
        if padding != 0:
            self.seq = nn.Sequential(
                nn.ReplicationPad2d(padding),
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                        padding=0, dilation=dilation, bias=bias),
                # nn.BatchNorm2d(out_channels)
            )
        else:
            self.seq = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                        padding=0, dilation=dilation, bias=bias),
                # nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return F.relu(self.seq(x))


class Inception_builder(nn.Module):
    """
    types of Inception block

    block_type: inception_block's type -- 'type1' or 'type2'
    in_channels: input channels
    b1_reduce: channel_1.1
    b1: channel_1.2
    b2_reduce: channel_2.1
    b2: channel_2.2
    b3: channel_3
    b4: channel_4
    """

    def __init__(self, block_type, in_channels, b1_reduce, b1, b2_reduce, b2, b3, b4):
        super(Inception_builder, self).__init__()
        self.block_type = block_type  # controlled by strings "type1", "type2"

        # 5x5 reduce, 5x5
        self.branch1_type1 = nn.Sequential(
            BN_Conv2d(in_channels, b1_reduce, 1, stride=1, padding=0, bias=False),
            BN_Conv2d(b1_reduce, b1, 5, stride=1, padding=2, bias=False)  # same padding
        )

        # 5x5 reduce, 2x3x3
        self.branch1_type2 = nn.Sequential(
            BN_Conv2d(in_channels, b1_reduce, 1, stride=1, padding=0, bias=False),
            BN_Conv2d(b1_reduce, b1, 3, stride=1, padding=1, bias=False),  # same padding
            BN_Conv2d(b1, b1, 3, stride=1, padding=1, bias=False)
        )

        # 3x3 reduce, 3x3
        self.branch2 = nn.Sequential(
            BN_Conv2d(in_channels, b2_reduce, 1, stride=1, padding=0, bias=False),
            BN_Conv2d(b2_reduce, b2, 3, stride=1, padding=1, bias=False)
        )

        # max pool, pool proj
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),  # to keep size, also use same padding
            BN_Conv2d(in_channels, b3, 1, stride=1, padding=0, bias=False)
        )

        # 1x1
        self.branch4 = BN_Conv2d(in_channels, b4, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        if self.block_type == "type1":
            out1 = self.branch1_type1(x)
            out2 = self.branch2(x)
        elif self.block_type == "type2":
            out1 = self.branch1_type2(x)
            out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat((out1, out2, out3, out4), 1)
        # out = torch.cat((x, out), 1)

        return out

class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.pad = nn.ReplicationPad2d(3 // 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(self.pad(x)))], 1)


class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)

    def forward(self, x):
        return x + self.lff(self.layers(x))  # local residual learning


class RDN(nn.Module):
    def __init__(self, scale_factor, num_channels, num_features, growth_rate, num_blocks, num_layers):
        super(RDN, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers

        self.pad = nn.ReplicationPad2d(3 // 2)

        # shallow feature extraction
        self.sfe1 = Inception_builder('type2', num_channels, 8, 16, 16, 32, 8, 8)
        self.sfe2 = Inception_builder('type2', num_features, 8, 16, 16, 32, 8, 8)

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G, self.G, self.C))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.ReplicationPad2d(3 // 2),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=0)
        )

        # up-sampling
        assert 2 <= scale_factor <= 8
        if math.modf(math.log2(scale_factor))[0] == 0:
            self.upscale = []
            for _ in range(int(math.log2(scale_factor))):
                self.upscale.extend([nn.ReplicationPad2d(3 // 2),
                                     nn.Conv2d(self.G0, self.G0 * (2 ** 2), kernel_size=3, padding=0),
                                     nn.PixelShuffle(2)])
            self.upscale = nn.Sequential(*self.upscale)
        else:
            self.upscale = nn.Sequential(
                nn.ReplicationPad2d(3 // 2),
                nn.Conv2d(self.G0, self.G0 * (scale_factor ** 2), kernel_size=3, padding=0),
                nn.PixelShuffle(scale_factor)
            )

        self.output = nn.Conv2d(self.G0, num_channels, kernel_size=3, padding=0)

    def forward(self, x):
        x = self.pad(x)
        sfe1 = self.sfe1(x)
        # sfe2 = self.pad(sfe1)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe1  # global residual learning
        x = self.upscale(x)
        x = self.pad(x)
        x = self.output(x)
        return x
