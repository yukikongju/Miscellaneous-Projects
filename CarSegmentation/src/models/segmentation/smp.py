"""
This file contains the Segmentation Model Pytorch model class built using 'segmentation_models_pytorch' library.
"""

import segmentation_models_pytorch as smp
from torch import nn


class SegmentationModelNetwork(nn.Module):

    def __init__(
        self, arch: str, encoder_name: str, encoder_weights: str, in_channels: int, out_classes: int
    ):
        super().__init__()
        self.arch = arch
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.out_classes = out_classes
        self.model = smp.create_model(
            arch=arch,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            out_classes=out_classes,
        )

    def forward(self, x):
        return self.model(x)
