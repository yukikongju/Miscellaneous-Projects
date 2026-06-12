"""
Usage:
    python3 wiggler.py 100
    python3 wiggler.py 100 --variance 0.5 --max-distance=150

"""

import argparse
import random
import time

from pyautogui import moveRel


def get_args():
    parser = argparse.ArgumentParser(description="Mouse Wiggler")
    parser.add_argument("intervals", type=int, help="base seconds between wiggles")
    parser.add_argument(
        "--variance",
        type=float,
        default=0.5,
        help="variance as fraction of intervals (default: 0.3)",
    )
    parser.add_argument(
        "--max-distance",
        type=int,
        default=100,
        help="max pixels to move per axis (default: 100)",
    )

    args = parser.parse_args()
    return args.intervals, args.variance, args.max_distance


def get_random_interval(base_interval, variance):
    variation = base_interval * variance
    return base_interval + random.uniform(-variation, variation)


def wiggle(max_distance):
    import math

    radius_x = random.randint(int(max_distance * 0.3), max_distance)
    radius_y = random.randint(int(max_distance * 0.3), max_distance)
    num_points = random.randint(10, 20)
    duration_per_segment = random.uniform(0.0000015, 0.000003)

    start_angle = random.uniform(0, 2 * math.pi)
    prev_x, prev_y = 0, 0

    for i in range(num_points):
        angle = start_angle + (2 * math.pi * i / num_points)
        x = int(radius_x * math.cos(angle))
        y = int(radius_y * math.sin(angle))
        dx = x - prev_x
        dy = y - prev_y
        moveRel(dx, dy, duration=duration_per_segment)
        prev_x, prev_y = x, y


if __name__ == "__main__":
    base_interval, variance, max_distance = get_args()
    while True:
        wiggle(max_distance)
        sleep_time = get_random_interval(base_interval, variance)
        time.sleep(max(sleep_time, 0.5))
