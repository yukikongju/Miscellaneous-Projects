import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


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


def array_to_wav(
    arr: np.ndarray, wav_path: str, sample_rate: int = 44100, frequency: int = 440
) -> None:
    """
    Convert flattened numpy array to .wav file
    """
    if arr.ndim != 1:
        arr = flatten_image_array(arr)

    amplitude = np.iinfo(np.int16).max
    wav = (amplitude * np.sin(2.0 * np.pi * frequency * arr)).astype(np.int16)
    wavfile.write(wav_path, sample_rate, wav)
    logging.info(f"Created wav file at {wav_path}")


def show_image_array(arr: np.ndarray) -> None:
    """
    Show image array
    """
    plt.imshow(arr)
    plt.show()
