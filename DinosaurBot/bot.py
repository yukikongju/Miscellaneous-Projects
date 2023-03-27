import cv2 
import time
import numpy
import random
import pyautogui
from PIL import Image

episode_scores = []
num_episodes = 500
start_time = time.time()

for i in range(num_episodes):

    # choose how much ahead to look 
    start_ahead = random.randint(80, 120)
    ahead = start_ahead

    # choose how much time before increasing look ahead
    start_speedup = random.randint(100, 210)

    # check if color needs to be inverted

    # starting game
    pyautogui.press('space')
    time.sleep(1)
    episode_start = time.time()

    while True: 
        last_time = time.time()
        image = pyautogui.screenshot(region=(200, 375, 800, 350))
        #  image.show()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)





