import csv

import numpy as np
from cached_property import cached_property
from skimage.transform import estimate_transform, rescale
import matplotlib.pyplot as plt


class Calibrator(object):
    """
    Generates a set of transformations based on coordinates in a pattern viewed
    from each camera angle. Use .from_csv() or .auto() to construct.
    """

    def __init__(self):
        self.images = None
        self.coordinates = None
        self.box_normal = np.array([[0, 100], [0, 0], [100, 0], [100, 100]])

    @classmethod
    def from_csv(cls, csv_path):
        """
        Construct a new calibrator using a CSV file of coordinates.
        :param csv_path:
        :return: a Calibrator with the coordinates loaded from the CSV
        """
        cal = cls()
        cal.coordinates = np.zeros((4, 4, 2), dtype=np.float)

        with open(csv_path, 'r+') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', skipinitialspace=True)
            header = next(reader)
            for camera, index, *coordinate in reader:
                cal.coordinates[int(camera), int(index)] = coordinate
        return cal

    @classmethod
    def auto(cls, images):
        """
        Construct a new calibrator using a set of images. The calibrator will attempt
        to calculate the coordinates for you.
        :param images: a list of paths to calibration pattern images
        :return: a Calibrator with the coordinates calculated from the images
        """
        raise NotImplementedError

    @cached_property
    def transforms(self):
        if self.coordinates is None:
            del self.__dict__['transforms']
            raise AttributeError(
                'You need to define coordinates first. Use Calibrator.from_csv() or '
                'Calbrator.auto() to construct a new calibrator.')
        else:
            return [estimate_transform('projective', self.box_normal, points) for points
                    in self.coordinates]

    def scale(self, scale):
        if self.images is not None:
            self.images = [rescale(image, scale) for image in self.images]
        if self.coordinates is not None:
            self.coordinates *= scale
        if 'transforms' in self.__dict__:
            del self.__dict__['transforms']

    def display(self):
        if self.images is None or self.coordinates is None:
            raise AttributeError(
                'You need to define coordinates and images first (either manually or '
                'using one of the constructors).')
        else:
            for i in range(len(self.images)):
                plot = plt.imshow(self.images[i])
                plt.scatter(self.coordinates[i][:, 0], self.coordinates[i][:, 1], c='r', s=20)
                plt.show()
