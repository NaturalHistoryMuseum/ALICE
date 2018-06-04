import numpy as np
import pandas as pd
from skimage.transform import rescale

from .view import View


class Calibrator(object):
    """
    Generates a set of transformations based on coordinates in a pattern viewed
    from each camera angle. Use .from_csv() or .auto() to construct.
    """

    def __init__(self):
        self.images = None
        self._views = []

    @classmethod
    def from_csv(cls, csv_path):
        """
        Construct a new calibrator using a CSV file of coordinates.
        :param csv_path:
        :return: a Calibrator with the coordinates loaded from the CSV
        """
        cal = cls()
        data = pd.read_csv(csv_path, sep=' ', skipinitialspace=True)
        data = data.pivot(index='Camera', columns='Point')
        cal._views += [View(cam, pd.DataFrame(points.swaplevel()).unstack().values.astype(
            np.float64)) for cam, points in data.iterrows()]
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

    def scale(self, scale):
        """
        Rescales (in-place) any images and views in the calibrator.
        :param scale: the scale factor
        """
        if self.images is not None:
            self.images = [rescale(image, scale) for image in self.images]
        for v in self._views:
            v.scale(scale)

    @property
    def views(self):
        """
        A dictionary of views/cameras keyed on their id.
        :return: dict
        """
        return {v.id: v for v in self._views}