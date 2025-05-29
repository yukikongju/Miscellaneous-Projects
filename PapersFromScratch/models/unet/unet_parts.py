import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Output = (W-K+2P) / S + 1

W=572x572
K=3
S=1
P=?

"""


class DoubleConv(nn.Module):
    """
    Two 3x3 convolutions (unpadded convolutions), each followed by
    a rectified linear unit (ReLU)

    Note:
    - The official code has padding=1 to keep shape the same
    - If Input is [B, C, H, W], then ouput is also [B, C, H, W]
    """

    def __init__(self, in_channels, out_channels, mid_channels=None) -> None:
        super().__init__()
        mid_channels = mid_channels if mid_channels else out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    2x2 max pooling operation with stride 2 for downsampling

    Notes:
    - Spatial Size halved; channels double
    - Ex: [B, 64, 572, 572] => [B, 128, 284, 284] => [B, 256, 143, 143] => ...

    """

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            #  nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels=in_channels, out_channels=out_channels),
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    """
    Every step in the expansive path consists of an upsampling of the
    feature map followed by a 2x2 convolution (“up-convolution”) that halves
    the number of feature channels, a concatenation with the correspondingly
    cropped feature map from the contracting path, and two 3x3 convolutions,
    each followed by a ReLU

    Notes:
    - Spatial Size Doubled; channels halved
    - Ex: [B, 256, 143, 143] => [B, 128, 284, 284] => [B, 64, 572, 572] => ...
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )  # 28x28 => 56x56
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
            #  self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Params
        ------
        x1: torch.tensor
            feature map of shape [B, C1, H1, W1]
        x2: torch.tensor
            skip connection from encoder of shape [B, C2, H2, W2]

        Notes:
        - x has shape [B, C1 + C2, H, W]
        """
        x1 = self.up(x1)

        # pad size mismatch between skip connection and feature map BCHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [
                diffX // 2, diffX - diffX // 2,  # left, rigth
                diffY // 2, diffY - diffY // 2,  # up, down
            ])

        # concat:
        x = torch.concat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    At the final layer a 1x1 convolution is used to map each 64-
    component feature vector to the desired number of classes
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
