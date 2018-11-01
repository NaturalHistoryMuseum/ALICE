import numpy as np
import pandas as pd
from cached_property import cached_property
from matplotlib import pyplot as plt
from skimage.feature import match_descriptors

from scalabel.models import Specimen
from scalabel.models.views import FeaturesView
from scalabel.models.logger import logger


class FeaturesSpecimen(Specimen):
    """
    Tries to find common features in the images of a specimen. Arbitrarily
    assigns a base view and matches the others to features in this view.
    :param specimen_id: the ID of the associated specimen
    :param views: a list of View objects (views of the specimen)

    """

    def __init__(self, specimen_id, views):
        super(FeaturesSpecimen, self).__init__(specimen_id, views)
        self.base_view = views[0]
        self._match_table = pd.DataFrame({
            self.base_view.position.id: list(range(len(self.base_view.descriptors)))
            })
        for v in views[1:]:
            self.match(v)

    @classmethod
    def from_specimen(cls, specimen):
        """
        Create a new instance of this class from another instance of a Specimen class or
        subclass.
        :param specimen: the specimen to transform

        """
        nkp = 4000

        def _make_fv(kp):
            views = [FeaturesView.from_view(v, kp).tidy() for v in specimen.views]
            min_kp = min([len(v.keypoints) for v in views])
            views = [v.tidy(n=min_kp) for v in views]
            return cls(specimen.id, views)

        while nkp < 6000:
            fv = _make_fv(nkp)
            if len(fv.global_matches) > 100:
                break
            nkp += 1000

        return fv

    def match(self, other):
        """
        Tries to find matching features in the base image and another image. Adds the
        matches to an internal pandas table.
        :param other: a FeaturesView object

        """
        matches = match_descriptors(self.base_view.descriptors, other.descriptors,
                                    cross_check=True)
        matches = pd.Series({m[0]: m[1] for m in matches}).reindex(
            self._match_table.index)
        self._match_table[other.position.id] = matches

    def get_matches(self, first, second):
        """
        Get the indices of the features/keypoints that match in two FeaturesView objects.
        Will only return features that are present in all images.
        :param first: FeaturesView object one
        :param second: FeaturesView object two
        :return: a (n, 2) shaped numpy array where column 1 contains indices for
                 features in object one and column 2 contains the corresponding
                 indices for features in object two

        """
        matches = self._match_table.dropna(0)[
            [first.position.id, second.position.id]].astype(int).values
        return matches

    def _common_keypoints(self, *others):
        """
        Get the keypoints (feature coordinates) for a variable number of FeaturesView objects.
        :param *others: FeaturesView objects
        :return: a 3D numpy array with coordinates for
        matching features in all the given images

        """
        matches = self._match_table.dropna(0)
        keypoints = []
        for other in others:
            indices = matches[other.position.id].astype(int).values
            # the coordinates have to be flipped for later processing, hence the ::-1
            keypoints.append(other.keypoints[indices, ::-1])
        return np.stack(keypoints, axis=1)

    @property
    def display(self):
        """
        Returns a display image (the view images with the matching features marked).
        :return: an image as a numpy array

        """
        fig, axes = plt.subplots(1, len(self.views),
                                 figsize=self._figsize(
                                     [(self.views[0].image, len(self.views))]),
                                 squeeze=True)
        for ax, view in zip(axes.ravel(), self.views):
            ax.imshow(view.grey)
            points = self._common_keypoints(view).reshape(-1, 2)[::-1]
            ax.plot(points[..., 0], points[..., 1], 'r+')
            ax.axis('off')
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.set(title=view.position.id)
        fig.tight_layout()
        fig.canvas.draw()
        img_array = np.array(fig.canvas.renderer._renderer)
        plt.close('all')
        return img_array

    @cached_property
    def global_matches(self, visualise=False):
        """
        Get coordinates for matching features in each of the input images.
        :param visualise: if True, show display image (for debugging)
        :returns: a 3D numpy array with coordinates for matching features in all the
                  given images

        """
        kp = self._common_keypoints(*self.views)
        if visualise:
            self.show()
        logger.debug(f'{len(kp)} common keypoints found')
        return kp
