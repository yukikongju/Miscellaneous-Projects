import torch.nn as nn

class Block(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=0,
                      stride=1),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)


class LeNet(nn.Module):

    """
    1 -> 6 -> 16 -> 120
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self):
        pass

