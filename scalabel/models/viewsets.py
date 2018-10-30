import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt

from .base import View, ViewSet
from .view_position import ViewPosition
from .views import LoadingView


class Label(ViewSet):
    """
    A collection of View objects representing a single label.
    :param specimen_id: the ID of the associated specimen
    :param views: a list of View objects (views of the label)

    """

    def __init__(self, specimen_id, views):
        super(Label, self).__init__(views)
        self.specimen_id = specimen_id
        self._image = None

    @classmethod
    def from_label(cls, label):
        """
        Create a new instance of this class from another instance of a Label class or
        subclass.
        :param label: the label to transform

        """
        return cls(label.specimen_id, label.views)

    @property
    def image(self):
        """
        A merged image of all views (parts in self._parts).
        :return: an image as a numpy array

        """
        if self._image is None:
            assert all([v.image.shape == self.views[0].image.shape for v in
                        self.views])
            self._image = np.median(np.stack([v.image for v in self.views]), axis=0)
        return self._image

    @property
    def display(self):
        """
        Returns a display image (the view images plus the merged image).
        :return: an image as a numpy array

        """
        nrow = 1
        ncol = len(self.views) + 1
        fig, axes = plt.subplots(nrows=nrow, ncols=ncol,
                                 figsize=(ncol * 6, nrow * 5),
                                 squeeze=True)
        for ax, (title, img) in zip(axes.ravel(),
                                    [(v.position.id, v.image) for v in self.views] + [
                                        ('combined', self.image)]):
            ax.imshow(img)
            ax.axis('off')
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.set(title=title)
        fig.tight_layout()
        fig.canvas.draw()
        img_array = np.array(fig.canvas.renderer._renderer)
        plt.close()
        return img_array


class Specimen(ViewSet):
    """
    A collection of View objects representing a single specimen.
    :param specimen_id: the ID of the associated specimen
    :param views: a list of View objects (views of the specimen)

    """

    def __init__(self, specimen_id, views):
        super(Specimen, self).__init__(views)
        self.id = specimen_id
        self.labels = []

    @classmethod
    def from_images(cls, specimen_id, image_files):
        """
        Create a new specimen instance by loading images from files to create views.
        :param specimen_id: the ID of the specimen
        :param image_files: paths to the images of this specimen
        :return: Specimen

        """
        views = [LoadingView(v, img) for v, img in image_files]
        return cls(specimen_id, views)

    @classmethod
    def from_specimen(cls, specimen):
        """
        Create a new instance of this class from another instance of a Specimen class or
        subclass.
        :param specimen: the specimen to transform

        """
        return cls(specimen.id, specimen.views)

    @property
    def display(self):
        """
        Returns a display image (the original view images, then one row for each label).
        :return: an image as a numpy array

        """
        nrow = len(self.labels) + 1
        ncol = len(self.views)
        viewfig, viewaxes = plt.subplots(ncols=ncol, figsize=(ncol * 6, 5))
        for ax, view in zip(viewaxes, self.views):
            ax.imshow(view.original)
            ax.axis('off')
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.set(title=view.position.id)
        viewfig.tight_layout()
        viewfig.canvas.draw()
        views = np.array(viewfig.canvas.renderer._renderer)

        fig, axes = plt.subplots(nrows=nrow,
                                 figsize=(ncol * 6, nrow * 5))
        if nrow == 1:
            axes = [axes]
        for ax, img in zip(axes, [views] + [l.display for l in self.labels]):
            ax.imshow(img)
            ax.axis('off')
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
        fig.tight_layout()
        fig.canvas.draw()
        img_array = np.array(fig.canvas.renderer._renderer)
        plt.close()
        return img_array


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
        fig, axes = plt.subplots(1, len(self.views),
                                 figsize=(len(self.views) * 2, len(self.views) // 2),
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
        plt.close()
        return img_array

    def __getitem__(self, item):
        try:
            return next(v for v in self.views if v.position.id == item)
        except StopIteration:
            raise KeyError
