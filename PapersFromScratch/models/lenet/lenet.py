import torch
import torch.nn as nn
from typing import List

class Block(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=0,
                      stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.layer(x)


class LeNet(nn.Module):

    """
    1 -> 6 -> 16 -> 120

    Input: (B, C=1, H=32, W=32)

    Note: 
    - The feature map on the last convolution layer needs to be included 
      in the fc_features because I haven't been able to compute it 
      programatically. In the paper, the last conv layer is 16x5x5, so 
      the fc_features = [400, 120, 84]
    """

    def __init__(self, in_channels: int, n_classes: int, conv_features: List[int],
                 fc_features: List[int]) -> None:
        super().__init__()
        self.conv_layers = nn.ModuleList()
        conv_features = [in_channels] + conv_features
        for idx in range(len(conv_features)-1):
            in_layer, out_layer = conv_features[idx], conv_features[idx+1]
            self.conv_layers.append(Block(in_layer, out_layer))

        self.fc_layers = nn.ModuleList()
        for idx in range(len(fc_features)-1):
            in_features, out_features = fc_features[idx], fc_features[idx+1]
            self.fc_layers.append(nn.Linear(in_features, out_features))
        self.fc_layers.append(nn.Linear(fc_features[-1], n_classes))


    def forward(self, x):
        out = x
        for layer in self.conv_layers:
            out = layer(out)
        out = out.flatten(start_dim=1)
        for layer in self.fc_layers:
            out = layer(out)
        return out

