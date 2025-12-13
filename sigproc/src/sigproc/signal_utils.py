import logging
import numpy as np
from scipy.io import wavfile
from . import image_utils
from typing import Tuple


def read_signal(wav_path: str) -> Tuple[int, np.ndarray]:
    """
    Reading signal from .wav file
    """
    try:
        sample_rate, data = wavfile.read(wav_path)
        print(sample_rate, data.shape)
    except Exception as e:
        raise FileNotFoundError(f"{wav_path} not found. Exiting with error {e}")
    return sample_rate, data


def array_to_wav(
    arr: np.ndarray, wav_path: str, sample_rate: int = 44100, frequency: int = 440
) -> None:
    """
    Convert flattened numpy array to .wav file
    """
    if arr.ndim != 1:
        arr = image_utils.flatten_image_array(arr)

    amplitude = np.iinfo(np.int16).max
    wav = (amplitude * np.sin(2.0 * np.pi * frequency * arr)).astype(np.int16)
    wavfile.write(wav_path, sample_rate, wav)
    logging.info(f"Created wav file at {wav_path}")
