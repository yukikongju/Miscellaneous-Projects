"""
This file contains the class to load the Car Dataset using PyTorch Dataset
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import Dataset


class CarDataset(Dataset):

    CLASSES = [
        "sky",
        "building",
        "pole",
        "road",
        "pavement",
        "tree",
        "signsymbol",
        "fence",
        "car",
        "pedestrian",
        "bicyclist",
        "unlabelled",
    ]

    def __init__(self, img_dir: str, annotation_dir: str, classes=None, augmentation=None):
        self.img_ids = os.listdir(img_dir)
        self.images_fps = [os.path.join(img_dir, image_id) for image_id in self.img_ids]
        self.masks_fps = [os.path.join(annotation_dir, image_id) for image_id in self.img_ids]
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.augmentation = augmentation

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        image = cv2.imread(self.images_fps[idx])
        # convert BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[idx], 0)  # (height, width, channels)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype("float")

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image.transpose(2, 0, 1), mask.transpose(2, 0, 1)  # (channels, height, width)
