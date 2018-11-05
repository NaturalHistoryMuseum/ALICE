from numba import njit
import numpy as np


@njit
def weighted_median(images, masks):
    """

    :param images:
    :param masks: 

    """
    height, width = images[0].shape[:2]

    out = np.zeros((height, width, 3))

    for i in range(height):
        for j in range(width):
            pixels = np.stack(tuple(image[i, j] for image, mask in zip(images, masks) if mask[i, j]))
            out[i, j] = np.median(pixels, axis=0) if pixels.size > 0 else 0

    return out
