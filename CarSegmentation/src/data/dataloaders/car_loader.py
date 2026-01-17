"""
This file contains the CarSegmentationDataLoader function used to load the CarSegmentationDataset into a DataLoader class
"""

from augmentations.car import get_training_augmentation, get_validation_augmentation
from data.datasets.car import CarSegmentationDataset
from schemas.dataloaders.car_config import CarSegmentationDataLoaderConfig
from utils.registry import register, DATALOADER_REGISTRY
from torch.utils.data import DataLoader


@register(DATALOADER_REGISTRY, "car_segmentation")
def get_car_segmentation_dataloader(cfg: CarSegmentationDataLoaderConfig):
    augmentation_dct = {
        "training": get_training_augmentation(),
        "validation": get_validation_augmentation(),
    }

    dataset = CarSegmentationDataset(
        cfg["images_dir"],
        cfg["annotations_dir"],
        classes=cfg["classes"],
        augmentation=augmentation_dct[cfg["augmentation"]],
    )
    dataloader = DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"]
    )
    return dataloader
