import pyautogui
import numpy as np
import cv2 as cv
import time 

def get_local_screenshot(x, y, w, h):
    """ 
    """
    screenshot = pyautogui.screenshot(region=(x, y, w, h))
    screenshot = np.array(screenshot)
    screenshot = cv.cvtColor(screenshot, cv.COLOR_RGB2BGR)
    cv.imshow('', screenshot)
    cv.waitKey()

def locate_pixel_on_screen(row, col):
    """ 
    """
    screenshot = pyautogui.screenshot()
    screenshot = np.array(screenshot)
    screenshot = cv.cvtColor(screenshot, cv.COLOR_RGB2BGR)
    for i in range(row-10, row+10):
        for j in range(col-10, col+10):
            screenshot[i][j] = (0, 0, 255)
    screenshot[row][col] = (255, 255, 255)
    cv.imshow('', screenshot)
    cv.waitKey()
    
# check tiles positions
#  locate_pixel_on_screen(650, 430)
#  locate_pixel_on_screen(650, 530)
#  locate_pixel_on_screen(650, 630)
#  locate_pixel_on_screen(650, 730)

tile1_coordinates = (430, 650)
tile2_coordinates = (530, 650)
tile3_coordinates = (630, 650)
tile4_coordinates = (730, 650)

def is_pixel_black(coordinates):
    """ 
    """
    screenshot = pyautogui.screenshot()
    red, green, blue = screenshot.getpixel(coordinates)
    if red < 25 and green < 5 and blue < 50: 
        return True
    return False
    

while True:
    if pyautogui.pixel(430, 700)[0] < 25:
        pyautogui.click(430, 700)
    if pyautogui.pixel(530, 700)[0] < 25:
        pyautogui.click(530, 700)
    if pyautogui.pixel(630, 700)[0] < 25:
        pyautogui.click(630, 700)
    if pyautogui.pixel(730, 700)[0] < 25:
        pyautogui.click(730, 700)

