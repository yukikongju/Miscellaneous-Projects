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
directions = ["top", "down", "right", "left"]
i = 0

while True:
    i += 1
    word = random.choice(directions)
    #  word = directions[i % len(directions)]
    print(word)
    os.system(f"say '{word}'")
    time.sleep(interval)
