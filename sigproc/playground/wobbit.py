#  https://scikit-image.org/docs/0.25.x/api/skimage.filters.html

import matplotlib.pyplot as plt
import skimage as ski
import numpy as np
from PIL import Image

# --- read image
file = "data/wobbit.jpeg"
img = Image.open(file)
img = np.array(img)

# --- sharpen the image
#  x = ski.filters.butterworth(img, high_pass=True)
img_sharpened = ski.filters.unsharp_mask(img, radius=5, amount=1.5)

# --- background / foreground separation: (1) canny + morphology (2) watershed
# -- (1) canny + morphology
img_gray = ski.color.rgb2gray(img)
edges = ski.feature.canny(img_gray, sigma=2)

#  edges = np.zeros_like(img, dtype=np.float32)
#  for i in range(3):
#      edges[:, :, i] = ski.feature.canny(img[:, :, i], sigma=1.2)
#  edges = 255.0 * edges
#  print(edges.shape)
#  print(edges.max())


# --- plot
fig, axs = plt.subplots(3, 1)
axs[0].imshow(img)
axs[1].imshow(img_sharpened)
axs[2].imshow(edges)

plt.show()
