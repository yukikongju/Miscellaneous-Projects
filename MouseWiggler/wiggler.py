import argparse
import time

from pyautogui import moveRel


def get_args():
    """ 
    Initialize parser to get args
    return: args
        - intervals: intervals between wiggle (in seconds)
        - duration: wiggle duration (in seconds)
    """
    parser = argparse.ArgumentParser(description='Mouse Wiggler')
    parser.add_argument('intervals', type=int, help='seconds between wiggle')
    parser.add_argument('duration', type=int, help='wiggle duration')

    args = parser.parse_args()

    intervals = args.intervals
    duration = args.duration

    return intervals, duration

def wiggle(duration):
    moveRel(-50, -50, duration=1)
    moveRel(50, 50, duration=duration)

if __name__ == "__main__":
    intervals, duration = get_args()
    while True:
        wiggle(duration)
        time.sleep(intervals)
    

