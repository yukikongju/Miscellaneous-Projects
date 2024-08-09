import numpy as np
import matplotlib.pyplot as plt


def get_dummy_circle_image(img_size=200, h=0, k=0) -> np.array:
    """
    return numpy array of circle
    (x-h)^2 + (y-k)^2 = r^2
    """
    img = np.zeros((img_size, img_size), dtype=int)

    r = img_size // 2
    cx, cy = img_size // 2, img_size // 2 

    for x in range(img_size):
        for y in range(img_size):
            if ((cx-x)**2 + (cy-y)**2) <= r**2:
                img[x, y] = 255

    return img

def apply_antialiasing(og_img: np.array, img_size: int = 200) -> np.array:
    """

    """
    img = og_img.copy()
    grid_size = 5

    r = img_size // 2
    cx, cy = img_size // 2, img_size // 2
    for i in range(img_size):
        for j in range(img_size):
            is_inside = False
            for x in range(grid_size):
                for y in range(grid_size):
                    sx = i + x / grid_size
                    sy = j + y / grid_size
                    if (cx-sx)**2 + (cy-sy)**2 <= r**2:
                        is_inside += 1

            img[i, j] = (is_inside / (grid_size * grid_size)) * 255

    return img


def main():
    img_size = 200
    img = get_dummy_circle_image(img_size=img_size)
    interpolation_methods = ['sinc', 'kaiser', 'antialiased',
         'hanning', 'lanczos', 'none', 'bilinear', 'bicubic', 'blackman',
         'spline16', 'gaussian', 'hamming', 'catrom', 'bessel', 'spline36', 
         'quadric', 'nearest', 'hermite', 'mitchell']


    # anti-aliasing the image
    antialiased_img = apply_antialiasing(og_img=img, img_size=img_size)

    # print images side by side
    fig, axes = plt.subplots(1, 2, figsize=(10,5))
    axes[0].imshow(img, cmap='grey', interpolation='antialiased')
    axes[1].imshow(antialiased_img, cmap='grey', interpolation='antialiased')

    plt.show()


if __name__ == "__main__":
    main()


