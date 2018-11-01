import numpy as np
import skimage.draw
import skimage.transform
from matplotlib import pyplot as plt

from .view_position import ViewPosition
from .logger import logger


class View(object):
    """
    An image of a specimen from a specified position.
    :param view_position: a defined position in the current calibration from which
                          this photo was taken
    :param image: the image as a numpy array
    :param original: the original image - preserved through view transformations where
                     self.image may be warped

    """

    def __init__(self, view_position: ViewPosition, image, original):
        self.position = view_position
        self._image = image
        self._original = original

    @property
    def image(self):
        """
        Purely for lazy-loading purposes.
        :return:
        """
        return self._image

    @property
    def original(self):
        """
        Also for lazy-loading.
        :return:
        """
        return self._original

    @classmethod
    def from_view(cls, view):
        """
        Create a new instance of this class from another instance of a View class or
        subclass.
        :param view: the view to transform

        """
        return cls(view.position, view.image, view.original)

    @classmethod
    def from_file(cls, view_position, imgfile):
        """
        Create a new instance of this class by loading an image directly from a file.
        :param view_position: a defined position in the current calibration from which
                              this photo was taken
        :param imgfile: path to the image file

        """
        img = plt.imread(imgfile)
        return cls(view_position, img, img)

    @property
    def display(self):
        """
        Returns a display image (the image with the view position marked on it).
        :return: an image as a numpy array

        """
        return self.position.display(self.image)


class ViewSet(object):
    """
    A collection of View objects.
    :param views: a list of View objects
    """

    def __init__(self, views):
        self.views = views

    @staticmethod
    def _figsize(rows):
        """
        Helper method to get a figure size for display images.
        :param rows: a list of tuples of (example_image, number_in_row)
        :return: (w,h)
        """
        max_width = max(i.shape[1] * n for i, n in rows)
        height = sum([i.shape[0] for i, n in rows])
        dpi = plt.rcParams['figure.dpi']
        if max_width / dpi > 100:
            ratio = height / max_width
            w = 24
            h = int(w * ratio)
        else:
            w = max_width / dpi
            h = height / dpi
        return w, h

    @property
    def display(self):
        """
        Returns a display image (all view images in a row, with titles).
        :return: an image as a numpy array

        """
        fig, axes = plt.subplots(1, len(self.views),
                                 figsize=self._figsize(
                                     [(self.views[0].image, len(self.views))]),
                                 squeeze=True)
        for ax, view in zip(axes.ravel(), self.views):
            ax.imshow(view.image)
            ax.axis('off')
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.set(title=view.position.id)
        fig.tight_layout()
        fig.canvas.draw()
        img_array = np.array(fig.canvas.renderer._renderer)
        plt.close('all')
        return img_array

    def show(self):
        """
        Show the display image. Just a helper method for debugging.

        """
        plt.imshow(self.display)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def save(self, fn):
        """
        Save the display image as a file.
        :param fn: the file name/path

        """
        plt.imsave(fn, self.display)


class MultipleTransformations:
    """
    Generate transformations for sets of keypoints.
    :param transform_type:
    :param num_angles:

    """

    def __init__(self, transform_type, num_angles):
        self.models = [transform_type() for _ in range(num_angles)]

    def estimate(self, points):
        """
        Estimate the transforms.
        :param points: the points to transform

        """
        for current_points, trans in zip(points.transpose(1, 0, 2)[:-1], self.models):
            trans.estimate(current_points, points[:, -1])
        return self

    def residual(self, points):
        """

        :param points: 

        """
        total = []
        for current_points, trans in zip(points.transpose(1, 0, 2)[:-1], self.models):
            distance = ((trans(current_points) - points[:, -1]) ** 2).sum(axis=1)
            total.append(np.minimum(200000, np.ceil(distance)).astype(np.int32))
        return np.stack(total).max(axis=0)

    @classmethod
    def initialise_model(cls, points):
        """

        :param points: 

        """
        num_points = points.shape[0]
        minimal_subset = points[np.random.choice(num_points, 8, replace=False)]
        return cls(skimage.transform.ProjectiveTransform,
                   num_angles=4).estimate(minimal_subset)
