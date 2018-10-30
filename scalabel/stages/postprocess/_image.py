from numba import njit
import numpy as np
import skimage


def improve_contrast(image, discard=5):
    """

    :param image: param discard: (Default value = 5)
    :param discard:  (Default value = 5)

    """
    image = skimage.img_as_float(image)
    out = np.zeros_like(image)

    for c, channel in enumerate(image.transpose(2, 0, 1)):
        low = np.percentile(channel.flatten(), discard)
        high = np.percentile(channel.flatten(), 100 - discard)
        out[..., c] = (np.clip(channel, low, high) - low) / (high - low)

    return out


@njit
def weighted_median(images, masks):
    """

    :param images: param masks:
    :param masks: 

    """
    height, width = images[0].shape[:2]

    out = np.zeros((height, width, 3))

    for i in range(height):
        for j in range(width):
            pixels = np.stack(tuple(image[i, j] for image, mask in zip(images, masks) if mask[i, j]))
            out[i, j] = np.median(pixels, axis=0) if pixels.size > 0 else 0

    return out
