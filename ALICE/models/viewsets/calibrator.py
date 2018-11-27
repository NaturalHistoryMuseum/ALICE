import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt

from ALICE.models.views import View
from ALICE.models.view_position import ViewPosition
from .base import ViewSet


class Calibrator(ViewSet):
    """
    Generates a set of transformations based on coordinates in a pattern viewed from
    each camera angle. Use a classmethod like .from_csv() or .graphical() to construct.

    """

    @classmethod
    def load_or_new(cls, csv_path, image_files):
        """
        If the given csv file exists, construct a calibrator from that; if not,
        use the graphical method to construct a new calibrator.
        :param csv_path: path to csv file
        :param image_files: images to use for constructing a new calibrator if the csv
                            does not exist
        :return: Calibrator

        """
        if os.path.exists(csv_path):
            return cls.from_csv(csv_path)
        else:
            return cls.graphical(image_files)

    @classmethod
    def from_csv(cls, csv_path):
        """
        Construct a new calibrator using a CSV file of coordinates.
        :param csv_path: path to the csv file
        :return: a Calibrator with the coordinates loaded from the CSV

        """
        data = pd.read_csv(csv_path, sep='\t', skipinitialspace=True)
        data = data.pivot(index='Camera', columns='Point')
        views = [View(
            ViewPosition(cam, pd.DataFrame(points.swaplevel()).unstack().values.astype(
                np.float64)), None, None) for cam, points in data.iterrows()]
        return cls(views)

    @classmethod
    def graphical(cls, images):
        """
        Get coordinates from user input on matplotlib graphs. Obviously not the most
        ideal way of doing this.
        :param images: a list of paths to calibration pattern images
        :return: a Calibrator

        """
        images = [(plt.imread(i), i.split(os.path.sep)[-1]) for i in images]
        views = [View(ViewPosition.click(img, path), img, img) for img, path in images]
        return cls(views)

    @classmethod
    def auto(cls, images):
        """
        Construct a new calibrator using a set of images. The calibrator will attempt
        to calculate the coordinates for you.
        :param images: a list of paths to calibration pattern images
        :return: a Calibrator with the coordinates calculated from the images

        """
        raise NotImplementedError

    def to_csv(self, csv_path):
        """
        Save the calibration to a csv file. This can be reloaded at a later point.
        :param csv_path: the path to save to

        """
        points = []
        for v in self.views:
            for i, xy in enumerate(v.position.coordinates):
                points.append([v.position.id, i, *xy])
        pd.DataFrame(data=points, columns=['Camera', 'Point', 'x', 'y']).to_csv(
            csv_path, sep='\t', index=False)

    @property
    def display(self):
        """
        Returns a display image (the display images for each of the views). This will
        only work if loaded from images because otherwise the views will not have any
        associated images.
        :return: an image as a numpy array

        """
        rows = [(self.views[0].display, len(self.views))]
        fig, axes = plt.subplots(1, len(self.views),
                                 figsize=self._figsize(rows),
                                 squeeze=True)
        for ax, view in zip(axes.ravel(), self.views):
            ax.imshow(view.display)
            ax.axis('off')
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.set(title=view.position.id)
        fig.tight_layout()
        fig.canvas.draw()
        img_array = np.array(fig.canvas.renderer._renderer)
        plt.close('all')
        return img_array

    def __getitem__(self, item):
        try:
            return next(v for v in self.views if v.position.id == item)
        except StopIteration:
            raise KeyError
