import numpy as np
import skimage


def improve_contrast(image, discard=2):
    """

    :param image:
    :param discard:

    """
    image = skimage.img_as_float(image)
    out = np.zeros_like(image)

    for c, channel in enumerate(image.transpose(2, 0, 1)):
        low = np.percentile(channel.flatten(), discard)
        high = np.percentile(channel.flatten(), 100 - discard)
        out[..., c] = (np.clip(channel, low, high) - low) / (high - low)

    return out
