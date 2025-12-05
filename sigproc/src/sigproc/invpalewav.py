import argparse
import logging
from . import image_utils, signal_utils


def run():
    parser = argparse.ArgumentParser(
        prog="invpalewav", description="converting .wav file to image"
    )
    parser.add_argument("-w", "--wav_path")
    parser.add_argument("-i", "--img_path")
    args = parser.parse_args()

    logging.basicConfig(format="[%(asctime)s] %(message)s", level=logging.INFO)

    print(args.img_path, args.wav_path)
