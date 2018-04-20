import numpy as np
import skimage.io
from skimage.morphology import skeletonize
from skimage.transform import hough_line, hough_line_peaks


def intersection(line1, line2):
    theta1, d1 = line1
    theta2, d2 = line2

    x = (np.sin(theta1) * d2 - np.sin(theta2) * d1) / np.sin(theta1 - theta2)
    y = (d1 - x * np.cos(theta1)) / np.sin(theta1)

    return x, y


colour = [('r', np.ubyte), ('g', np.ubyte), ('b', np.ubyte)]

with open('square.csv', 'w+') as csvfile:
    print('Camera Point x    y   ', file=csvfile)

    image = np.rec.array(skimage.io.imread('../data/view1.png'), dtype=colour).squeeze()

    c1 = (image.r > 128) & (image.g < 128) & (image.b < 128)
    c2 = (image.r < 128) & (image.g > 128) & (image.b < 128)
    c3 = (image.r < 128) & (image.g < 128) & (image.b > 128)
    c4 = (image.r > 128) & (image.g < 128) & (image.b > 128)

    for point_index, corner in enumerate([c1, c2, c3, c4]):
        h, theta, d = hough_line(skeletonize(corner))
        _, theta_peak, d_peak = hough_line_peaks(h, theta, d, threshold=(0.25 * h.max()), num_peaks=2)

        lines = zip(theta_peak, d_peak)
        x, y = intersection(*lines)

        print(f'{0:<6} {point_index:<5} {int(x):<4} {int(y):<4}', file=csvfile)
