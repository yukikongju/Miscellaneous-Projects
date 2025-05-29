import torch
import torch.nn as nn

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

    Note: The official code has padding=1
    """

    def __init__(self, in_channels, out_channels, mid_channels=None) -> None:
        super().__init__()
        mid_channels = mid_channels if mid_channels else out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=0,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=0,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)
