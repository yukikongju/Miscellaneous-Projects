import numpy as np
import scipy

from . import signal_utils

file_path = "/data/.wav"
fs, audio = signal_utils.read_signal(file_path)
