import itertools

from matplotlib import pyplot as plt
import numpy as np
import skimage.color
import skimage.io
from operator import itemgetter
from pathlib import Path
from skimage.morphology import closing, square, thin
from skimage.transform import hough_line, hough_line_peaks, rescale

from scalabel.stages.postprocess._image import improve_contrast
from scalabel.models.logger import logger


class Point:
    """ """
    def __init__(self, location, image, mask):
        self.location = location
        self.image = image
        self.mask = mask

    @property
    def colour(self):
        """ """
        x, y = self.location
        neighbourhood = (slice(int(y) - 40, int(y) + 40),
                         slice(int(x) - 40, int(x) + 40))
        return np.array([(channel[neighbourhood] * self.mask[neighbourhood]).sum() /
                         self.mask[neighbourhood].sum()
                         for channel in self.image.transpose(2, 0, 1)])


def intersection(line1, line2):
    """

    :param line1: param line2:
    :param line2: 

    """
    theta1, d1 = line1
    theta2, d2 = line2

    x = (np.sin(theta1) * d2 - np.sin(theta2) * d1) / np.sin(theta1 - theta2)
    y = (d1 - x * np.cos(theta1)) / np.sin(theta1)

    return x, y


def visualise_lines(image, lines):
    """

    :param image: param lines:
    :param lines: 

    """
    height, width = image.shape[:2]
    plt.imshow(image)
    for theta, d in lines:
        x = np.arange(width)
        y = (d - x * np.cos(theta)) / np.sin(theta)
        plt.plot(x, y)
    plt.show()


def visualise_corners(image, corners):
    """

    :param image: param corners:
    :param corners: 

    """
    plt.imshow(image)
    for (x, y), colour in zip(corners, ('r', 'g', 'b', 'm')):
        plt.plot(x, y, colour + 'o')
    plt.show()


def normalise(a):
    """

    :param a: 

    """
    return (a - a.min()) / (a - a.min()).sum()


def colour_difference(points):
    """

    :param points: 

    """
    target_colours = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1)]

    colours = np.array([point.colour for point in points])
    return ((normalise(colours) - np.array(target_colours)) ** 2).sum()


def detect_corners(image, scale_factor):
    """

    :param image: param scale_factor:
    :param scale_factor: 

    """
    image = improve_contrast(rescale(image, 1 / scale_factor), discard=0.5)
    hue, saturation, value = skimage.color.rgb2hsv(image).transpose(2, 0, 1)
    mask = closing((saturation > 0.25) & (value > 0.5), selem=square(5))
    structure = thin(mask)
    h, theta, d = hough_line(structure)
    _, theta_peak, d_peak = hough_line_peaks(h, theta, d, min_distance=20,
                                             threshold=(0.1 * h.max()), num_peaks=4)

    lines = sorted(zip(theta_peak, d_peak), key=itemgetter(0))
    perpendicular_pairs = lines[:2], lines[2:]

    points = [Point(intersection(*lines), image, mask) for lines in
              itertools.product(*perpendicular_pairs)]
    return min(itertools.permutations(points), key=colour_difference)


image_names = ['0002_ALICE3.JPG', '0002_ALICE4.JPG', '0002_ALICE5.JPG',
               '0002_ALICE6.JPG']
images = [skimage.io.imread(Path() / 'data' / 'calibration' / 'pattern' / name) for name
          in image_names]

scale_factor = 5
corners = [detect_corners(image, scale_factor) for image in images]

with open('square.csv', 'w+') as csvfile:
    logger.debug('Camera Point x    y   ', file=csvfile)

    for image_index, image_corners in enumerate(corners):
        for point_index, point in enumerate(image_corners):
            x, y = np.array(point.location) * scale_factor
            logger.debug(f'{image_index:<6} {point_index:<5} {int(x):<4} {int(y):<4}',
                  file=csvfile)
