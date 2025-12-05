import logging
import numpy as np
from scipy.io import wavfile
from . import image_utils


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


