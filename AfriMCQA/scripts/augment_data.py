import numpy as np
import pandas as pd
from PIL import Image
from datasets import load_dataset
import albumentations as A


## ---- LOAD DATASET
LANGUAGES = ["Lingala", "Akan_Twi", "Hausa", "Kinyarwanda"]
COLUMNS = [
    "ID",
    "Country",
    "Language",
    "Category",
    "self_made",
    "eng_question",
    "native_question",
    "correct_en",
    "wrong_en_o1",
    "wrong_en_o2",
    "wrong_en_o3",
    "correct_native",
    "wrong_native_o1",
    "wrong_native_o2",
    "wrong_native_o3",
    "image",
]

lang = LANGUAGES[2]
dataset = load_dataset(
    "Atnafu/Afri-MCQA",
    f"{lang}_dev",
    split="dev",
    columns=COLUMNS,
)

## --- TODO: Filter specific question ID


## ---- AUGMENT DATASET


def get_augmented_dataset(dataset, transform):
    def augment_example(example):
        image = example["image"]
        if image is not None:
            img_array = np.array(image.convert("RGB"))
            augmented = transform(image=img_array)
            example["image"] = Image.fromarray(augmented["image"])
        return example

    return dataset.map(augment_example)


transform = A.Compose(
    [
        # Geometric Augmentation
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30, p=0.5),
        A.RandomCrop(height=224, width=224, p=0.3),
        # Photometric Augmentation
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    ]
)

dataset_augmented = get_augmented_dataset(dataset, transform)
dataset_augmented

## ---- Combine Dataset
