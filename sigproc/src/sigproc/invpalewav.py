import argparse
import matplotlib.pyplot as plt
import numpy as np
import logging
from . import image_utils, signal_utils


def run():
    parser = argparse.ArgumentParser(
        prog="invpalewav", description="converting .wav file to image"
    )
    parser.add_argument("-w", "--wav_path")
    #  parser.add_argument("-i", "--img_path")
    args = parser.parse_args()

    logging.basicConfig(format="[%(asctime)s] %(message)s", level=logging.INFO)

    sample_rate, signal = signal_utils.read_signal(args.wav_path)
    n = np.sqrt(len(signal)).astype(np.int16)
    print(n)
    x = signal.reshape(n, -1)
    print(x.shape)
    plt.imshow(x)
    plt.show()
