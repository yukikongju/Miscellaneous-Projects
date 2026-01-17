"""
This file contains the Segmentation Model Pytorch model class built using 'segmentation_models_pytorch' library.
"""

import segmentation_models_pytorch as smp
from torch import nn
from utils.registry import register, MODEL_REGISTRY
from schemas.models.segmentations.smp_config import SegmentationModelNetworkConfig


@register(MODEL_REGISTRY, "smp")
class SegmentationModelNetwork(nn.Module):

    def __init__(self, cfg: SegmentationModelNetworkConfig):
        super().__init__()
        self.name = cfg["name"]
        self.arch = cfg["arch"]
        self.encoder_name = cfg["encoder_name"]
        self.encoder_weights = cfg["encoder_weights"]
        self.in_channels = cfg["in_channels"]
        self.out_classes = cfg["out_classes"]
        self.model = smp.create_model(
            arch=self.arch,
            encoder_name=self.encoder_name,
            encoder_weights=self.encoder_weights,
            in_channels=self.in_channels,
            out_classes=self.out_classes,
        )

    def forward(self, x):
        return self.model(x)
