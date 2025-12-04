import argparse
import logging
from . import image_utils, signal_utils


def run():
    parser = argparse.ArgumentParser(
        prog="palewav", description="converting image to .wav file"
    )
    parser.add_argument("-i", "img_path")
    parser.add_argument("-w", "wav_path")
    args = parser.parse_args()

    logging.basicConfig(format="[%(asctime)s] %(message)s", level=logging.INFO)

    #  img_path = "data/lenna.png"
    #  wav_path = "outputs/lenna.wav"

    img = image_utils.read_image(args.img_path)
    arr = image_utils.image_to_grayscale_array(img, mode="L")
    signal_utils.array_to_wav(arr, args.wav_path)
