from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np


# Only for Train and Validation
class CloudDataset(Dataset):
    CLASSES = ["Fish", "Flower", "Gravel", "Sugar"]
    IMG_SIZE = (240, 360)  # original size: 1400x2100

    def __init__(self, dataset_path: str, split: str, train_split: float = 0.8) -> None:
        self.dataset_path = dataset_path
        self.split = split
        self.train_split = train_split

        self.df_train, self.image_ids = self._load_train_csv()

    def _load_train_csv(self):
        csv_path = self.dataset_path + "train.csv"
        try:
            df_train = pd.read_csv(csv_path)
        except Exception as e:
            raise FileNotFoundError(f"Couldn't find {csv_path}. Please check {e}")

        df_train[["Image", "Label"]] = df_train["Image_Label"].str.split("_", expand=True)
        image_ids = df_train["Image"].unique()
        n = len(image_ids)
        e_train = int(n * self.train_split)
        if self.split == "train":
            return df_train, image_ids[:e_train]
        elif self.split == "val":
            return df_train, image_ids[e_train:]
        else:
            raise ValueError(f"Split not implemented. Please select 'train' or 'test'")

    def __len__(self):
        return len(self.image_ids)

    def _get_mask(self, idx, shape=(1400, 2100)):
        """
        Masks: 'Fish', 'Flower', 'Gravel', 'Sugar',
        """
        height, width = shape
        file_name = self.image_ids[idx]
        masks = []
        for label in ["Fish", "Flower", "Gravel", "Sugar"]:
            image_mask = self.df_train["Image"] == file_name
            label_mask = self.df_train["Label"] == label
            encoded_pixels = self.df_train.loc[image_mask & label_mask, "EncodedPixels"].values[0]

            if pd.isna(encoded_pixels):
                masks.append(np.zeros((width, height), dtype=np.uint8).T)
                # masks.append(np.zeros((width, height), dtype=np.uint8))
                continue

            encoded_pixels = list(map(int, encoded_pixels.split()))
            mask = np.zeros(width * height, dtype=np.uint8)
            for i in range(0, len(encoded_pixels), 2):
                start, length = encoded_pixels[i] - 1, encoded_pixels[i + 1]
                mask[start : start + length] = 1
            mask = mask.reshape((width, height)).T
            # mask = mask.reshape((width, height))
            masks.append(mask)
        return np.stack(masks, axis=0)

    def __getitem__(self, idx):
        sub_directory = "train_images/"
        image = Image.open(self.dataset_path + sub_directory + self.image_ids[idx])
        image = image.resize(self.IMG_SIZE, Image.BILINEAR)
        img_array = np.array(image)
        mask = self._get_mask(idx)
        mask = np.stack(
            [np.array(Image.fromarray(m).resize(self.IMG_SIZE, Image.NEAREST)) for m in mask],
            axis=0,
        )

        if img_array.ndim == 4:
            # batch
            img_array = np.transpose(img_array, (0, 3, 1, 2))
        elif img_array.ndim == 3:
            # single image
            img_array = np.transpose(img_array, (2, 0, 1))
        else:
            raise ValueError(f"Unexpected shape: {img_array.shape}")

        # return {"image": img_array, "mask": mask} # np.transpose(img_array, (0, 3, 1, 2))
        return {"image": img_array, "mask": mask}  #
