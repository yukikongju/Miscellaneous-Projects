import logging
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
