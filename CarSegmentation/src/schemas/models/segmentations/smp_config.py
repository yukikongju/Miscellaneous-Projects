"""
This file contains the pydantic class to verify the SegmentationModelNetwork Config.
"""

from pydantic import BaseModel


class SegmentationModelNetworkConfig(BaseModel):
    name: str
    arch: str
    encoder_name: str
    encoder_weights: str
    in_channels: int
    out_classes: int
