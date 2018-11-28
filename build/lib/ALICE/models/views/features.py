import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.feature import ORB

from ALICE.models.utils import logger
from .base import View


class FeaturesView(View):
    """
    A view of a specimen or area of the specimen with an associated feature detector.
    :param view_position: a defined position in the current calibration from which
                          this photo was taken
    :param image: the image as a numpy array
    :param original: the original image - preserved through view transformations where
                     self.image may be warped
    :param nkp: the number of keypoints to find - if None, no limit/target.

    """
    nkp = 4000

    def __init__(self, view_position, image, original, nkp=None):
        super(FeaturesView, self).__init__(view_position, image, original)
        self.grey = rgb2gray(self.image)
        detector_args = {
            'n_keypoints': nkp
            } if nkp is not None else {}
        self.detector = ORB(**detector_args)
        self.detector.detect_and_extract(self.grey)
        self.descriptors = self.detector.descriptors
        self.keypoints = self.detector.keypoints
        logger.debug(f'found {len(self.keypoints)} keypoints in view {self.position.id}')

    @classmethod
    def from_view(cls, view, nkp=None):
        """
        Create a new instance of this class from another instance of a View class or
        subclass. Uses the class-defined number of keypoints if not given.
        :param view: the view to transform
        :param nkp: the number of keypoints to find

        """
        return cls(view.position, view.image, view.original, nkp=nkp or cls.nkp)

    def tidy(self, n=None):
        """
        Get rid of keypoints around the edges.
        :param n: the maximum number of keypoints to keep - if None, just remove
                  whatever is necessary.

        """
        nkp = len(self.keypoints)
        indices = []
        for ki, k in enumerate(self.keypoints):
            i, j = k.astype(int)
            limit = 5
            region = self.image[i - limit:i + limit + 1, j - limit:j + limit + 1]
            if (region.sum(axis=2) > 0).all():
                indices.append(ki)
        self.keypoints = self.keypoints[np.array(indices)]
        self.descriptors = self.descriptors[np.array(indices)]
        centrepoint = self.keypoints.mean(axis=0)
        distances = abs(self.keypoints - centrepoint)
        indices = np.argsort((distances * distances.std(axis=0)).max(axis=1))
        if n is not None:
            indices = indices[:n]
        self.keypoints = self.keypoints[indices]
        self.descriptors = self.descriptors[indices]
        logger.debug(f'removed {nkp - len(self.keypoints)} '
                     f'keypoints from view {self.position.id}')
        return self

    @property
    def display(self):
        """
        Returns a display image (the image with the keypoints plotted on it).
        :return: an image as a numpy array

        """
        fs = tuple([i / 200 for i in self.image.shape[:2]])
        fig, ax = plt.subplots(figsize=fs)
        ax.imshow(self.grey)
        ax.plot(self.keypoints[..., 1], self.keypoints[..., 0], 'r+')
        ax.axis('off')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.autoscale(tight=True)
        fig.tight_layout()
        fig.canvas.draw()
        img_array = np.array(fig.canvas.renderer._renderer)
        plt.close('all')
        return img_array
