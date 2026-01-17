"""
This file contains the pydantic class to verify the CarSegmentationDataset configurations.
"""

from pydantic import BaseModel
from typing import List


class CarSegmentationDataLoaderConfig(BaseModel):
    images_dir: str
    annotations_dir: str
    classes: List[str]
    augmentation: str
    batch_size: int
    num_workers: int
