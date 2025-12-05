import logging
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def read_image(img_path: str) -> Image.Image:
    """
    Read image from jpg/png file
    """
    try:
        img = Image.open(img_path)
        logging.info(f"Read image file {img_path} succesfully!")
        return img
    except Exception as e:
        raise FileNotFoundError(f"{img_path} not found. Exited with error: {e}")


def image_to_grayscale_array(img: Image.Image, mode: str = "L") -> np.ndarray:
    """
    Convert image to grayscale using 'luminance' strategy

    Params
    ------
    mode: str
        > 'L', 'I'
    """
    return np.array(img.convert(mode))


def flatten_image_array(arr: np.ndarray) -> np.ndarray:
    """
    Flatten image array to 1D

    RGB: [R values], [G values], [B values]

    """
    if arr.ndim == 1:
        logging.info("Image array already flat. Skipping...")
        return arr
    if arr.ndim >= 4:
        raise ValueError("Array doesn't represent image. Too many channels")
    return arr.reshape(-1)

def show_image_array(arr: np.ndarray) -> None:
    """
    Show image array
    """
    plt.imshow(arr)
    plt.show()
