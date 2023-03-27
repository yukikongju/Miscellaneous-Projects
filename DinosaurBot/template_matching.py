import pyautogui
import cv2 as cv
import numpy as np
import os 
import matplotlib.pyplot as plt

from PIL import Image

# Load templates and target images
template_path = "/home/yukikongju/Projects/Miscellaneous-Projects/DinosaurBot/templates/template_rod.png"
target_path = "/home/yukikongju/Projects/Miscellaneous-Projects/DinosaurBot/images/pond1.png"

template = cv.imread(template_path, cv.IMREAD_GRAYSCALE)
target = cv.imread(target_path, cv.IMREAD_GRAYSCALE)

methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR', 
           'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

# Perform template Matching
result = cv.matchTemplate(target, template, cv.TM_CCORR_NORMED)
plt.imshow(result, cmap = 'gray')
#  print(result)
plt.show()


# Get the position of the highest correlation
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

# Get the top-left and bottom-right coordinates of the matching region
top_left = max_loc
bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])

# Draw a rectangle around the matching region
cv.rectangle(target, top_left, bottom_right, (255, 0, 0), 2)

# Show the result
cv.imshow('Result', target)
cv.waitKey(0)
cv.destroyAllWindows()
