import pyautogui 
import time
import numpy as np
import cv2 as cv
from pynput.keyboard import Key, Listener


# color of center: (255, 219, 195)

# get screenshot of aimbooster 
#  screenshot = pyautogui.screenshot(region=(195, 325, 675, 500))
#  screenshot = np.array(screenshot)
#  screenshot = cv.cvtColor(screenshot, cv.COLOR_RGB2BGR)
#  cv.imshow('', screenshot)
#  cv.waitKey()

while True:
    flag = 0
    screenshot = pyautogui.screenshot(region=(195, 325, 675, 500))

    width, height = screenshot.size

    # click 
    for x in range(0, width, 5):
        for y in range(0, height, 5):
            r, g, b = screenshot.getpixel((x, y))
            if r == 255 and g == 219 and b == 195:
                flag = 1
                pyautogui.click(x + 195, y + 325)
                time.sleep(0.05)

            if flag == 1:
                break

#      # exit program
#      with Listener(on_press = exit_key) as listerner:
#          listener.join()

#  def exit_key(key):
#      if key == Key.esc: 
#          return False
    

