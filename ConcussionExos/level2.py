"""
Script to launch exercice for my concussion:)

Say aloud one of 4 directions ('top', 'down', 'left', 'right')
at a given pace (in bpm)

"""

import time
import random
import os


bpm = 60
interval = 60.0 / bpm
directions = ["one", "two", "three", "four"]
hands = ["left", "right"]
i = 0

while True:
    i += 1
    direction = random.choice(directions)
    hand = random.choice(hands)
    print(f"{i} => {hand} {direction}")
    os.system(f'say "{hand} {direction}"')
    time.sleep(interval)
