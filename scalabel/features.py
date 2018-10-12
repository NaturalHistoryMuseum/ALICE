import itertools
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.color import rgb2gray
from skimage.feature import ORB, match_descriptors


class ImageFeatures(object):
    """
    Describes an image and detects features in it using scikit's ORB detector.
    :param ix: index or other unique identifier for the image
    :param image: the image object as a numpy array
    """

    nkp = 3000

    def __init__(self, ix, image):
        self.ix = ix
        self.image = image
        self.grey = rgb2gray(self.image)
        self.detector = ORB(n_keypoints=self.nkp)
        self.detector.detect_and_extract(self.grey)
        self.descriptors = self.detector.descriptors
        self.keypoints = self.detector.keypoints
        self.remove_keypoints()

    def remove_keypoints(self):
        """
        Get rid of keypoints around the edges.
        """
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
        indices = np.argsort((distances * distances.std(axis=0)).max(axis=1))[
                  :self.nkp - 100]
        self.keypoints = self.keypoints[indices]
        self.descriptors = self.descriptors[indices]


class FeatureMatcher(object):
    """
    Given a group of images, finds matching image features between them. Arbitrarily
    assigns a base image and matches the others to features in this image,
    only returning the features that match in all images.
    :param images: ImageFeatures objects to match
    """

    def __init__(self, images):
        self.base_image = images[0]
        self.images = images
        self._match_table = pd.DataFrame({
            self.base_image.ix: list(range(len(self.base_image.descriptors)))
            })
        for img in images[1:]:
            self.match(img)

    def match(self, other):
        """
        Tries to find matching features in the base image and another image. Adds the
        matches to an internal pandas table.
        :param other: an ImageFeature object
        """
        matches = match_descriptors(self.base_image.descriptors, other.descriptors,
                                    cross_check=True)
        matches = pd.Series({m[0]: m[1] for m in matches}).reindex(
            self._match_table.index)
        self._match_table[other.ix] = matches

    def get_matches(self, first, second):
        """
        Get the indices of the features/keypoints that match in two ImageFeatures
        objects. Will only return features that are present in all images.
        :param first: ImageFeatures object one
        :param second: ImageFeatures object two
        :return: a (n, 2) shaped numpy array where column 1 contains indices for
                 features in object one and column 2 contains the corresponding indices
                 for features in object two
        """
        matches = self._match_table.dropna(0)[[first.ix, second.ix]].astype(int).values
        return matches

    def get_keypoints(self, *others):
        """
        Get the keypoints (feature coordinates) for a variable number of ImageFeatures
        objects.
        :param others: ImageFeatures object(s)
        :return: a 3D numpy array with coordinates for matching features in all the
                 given images
        """
        matches = self._match_table.dropna(0)
        keypoints = []
        for other in others:
            indices = matches[other.ix].astype(int).values
            # the coordinates have to be flipped for later processing, hence the ::-1
            keypoints.append(other.keypoints[indices, ::-1])
        return np.stack(keypoints, axis=1)

    def visualise(self):
        """
        Display the images with the matching features marked.
        """
        ncols = 2
        fig, axes = plt.subplots(nrows=int(len(self.images) / 2), ncols=ncols)
        for r, c, img in zip(itertools.count(0, 1 / ncols),
                             itertools.cycle(list(range(ncols))), self.images):
            r = int(math.floor(r))
            axes[r, c].imshow(img.image)
            points = self.get_keypoints(img).reshape(-1, 2)[::-1]
            axes[r, c].plot(points[..., 0], points[..., 1], 'r+')
            axes[r, c].axis('off')
        plt.show()

    @classmethod
    def global_matches(cls, images, visualise=False):
        """
        Get coordinates for matching features in each of the input images.
        :param visualise: boolean - display the images with the features marked if True
        :param images: a list of image objects (as numpy arrays)
        :return: a 3D numpy array with coordinates for matching features in all the
                 given images
        """
        image_features = [ImageFeatures(ix, img) for ix, img in enumerate(images)]
        fm = cls(image_features)
        kp = fm.get_keypoints(*image_features)
        if visualise:
            fm.visualise()
        return kp
