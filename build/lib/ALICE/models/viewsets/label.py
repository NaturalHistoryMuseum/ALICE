import pyflow

import numpy as np
import skimage
from matplotlib import pyplot as plt
from skimage.measure import ransac

from ALICE.models.utils import SimilarAsPossible
from ALICE.models.views import WarpedView
from ALICE.models.viewsets import FeatureComparer
from ALICE.utils.image import improve_contrast
from .base import ViewSet


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
        rows = [(self.views[0].image, len(self.views) + 1)]
        fig, axes = plt.subplots(nrows=nrow, ncols=ncol,
                                 figsize=self._figsize(rows),
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
        plt.close('all')
        return img_array

    def save(self, fn):
        """
        Save the combined image as a file.
        :param fn: the file name/path

        """
        plt.imsave(fn, self.image)


class AlignedLabel(Label):
    """
    A collection of View objects representing a single label. The view images are
    warped to align with each other.
    :param specimen_id: the ID of the associated specimen
    :param views: a list of View objects (views of the label)

    """

    def __init__(self, specimen_id, views):
        super(AlignedLabel, self).__init__(specimen_id, views)
        self.comparer = FeatureComparer.ensure_minimum(specimen_id, views, 10, 500)
        # align homography (using FeaturesView objects)
        self.views = [self._align_homography(bv, v) for bv, v in
                      zip(self.comparer.views, np.roll(self.comparer.views, -1))]
        # self.views[0] = WarpedView(self.views[0].position, self.views[0].image,
        #                            self.views[0].original)
        self.views[1:] = [self._regularised_flow(v) for v in self.views[1:]]

    def _align_homography(self, base_view, view):
        """
        Match features between the view and the base view, generate transform
        parameters, and warp the view image to align with the base view. Uses the
        RANSAC (random sample consensus) method.
        :param view: the view to compare to the base view and warp as necessary

        """
        # matches = match_descriptors(view.descriptors,
        #                             self.views[0].descriptors, cross_check=True)
        matches = self.comparer.get_matches(base_view, view)
        src = view.keypoints[matches[:, 0], ::-1]
        dst = base_view.keypoints[matches[:, 1], ::-1]

        warped = np.array([np.nan])
        while np.isnan(warped.sum()):
            h, inliers = ransac([src, dst], skimage.transform.ProjectiveTransform,
                                min_samples=8, residual_threshold=2, max_trials=400)

            warped = skimage.transform.warp(view.image, h.inverse)
        return WarpedView(view.position, warped, view.original)

    def _regularised_flow(self, view):
        """
        Find and correct inconsistencies between the given view and the base view.
        :param view: the view to compare to the base view

        """
        flow = self._optical_flow(view.image)
        flow_reverse = self._optical_flow(view.image, reverse=True)

        height, width = view.image.shape[:2]
        grid = np.stack(np.mgrid[:height, :width][::-1], axis=2)

        flow = flow[10:-10]
        flow_reverse = flow_reverse[10:-10]
        grid = grid[10:-10]

        p = grid.reshape(-1, 2) + 0.5
        p_hat = (grid - flow).reshape(-1, 2) + 0.5

        valid = (self.bidirectional_similarity(flow, flow_reverse) < 2).flatten()

        s = SimilarAsPossible(shape=view.image.shape, grid_separation=(40, 40)).fit(
            p[valid], p_hat[valid], alpha=2)

        warped = np.ma.masked_invalid(
            skimage.transform.warp(view.image, s.transformation.inverse,
                                   cval=np.nan)).filled(fill_value=0)

        return WarpedView(view.position, warped, view.original)

    def _optical_flow(self, image, reverse=False):
        """
        Find inconsistencies ('flow') between an image and the base image
        :param image: the image to compare to the base image
        :param reverse: if True, use the given image as the base

        """
        target = skimage.img_as_float(
            image if reverse else self.views[0].image)
        other = skimage.img_as_float(self.views[0].image if reverse else image)
        # Flow Options:
        alpha = 0.012
        ratio = 0.75
        min_width = 20
        n_outer_fp_iterations = 7
        n_inner_fp_iterations = 1
        n_sor_iterations = 30
        # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
        col_type = 0

        u, v, im2_w = pyflow.coarse2fine_flow(
            target.copy(order='C'), other.copy(order='C'),
            alpha, ratio, min_width, n_outer_fp_iterations, n_inner_fp_iterations,
            n_sor_iterations, col_type)

        return np.stack((u, v), axis=2)

    @property
    def display(self):
        """
        Returns a display image (the original view images plus the merged image,
        and the aligned view images plus the merged image).
        :return: an image as a numpy array

        """
        nrow = 2
        ncol = len(self.views) + 1
        rows = [(self.views[0].original, len(self.views)),
                (self.views[0].image, len(self.views) + 1)]
        fig, axes = plt.subplots(nrows=nrow, ncols=ncol,
                                 figsize=self._figsize(rows),
                                 squeeze=True)
        originals = [(v.position.id, v.original) for v in self.views] + [
            ('combined', np.median(np.stack([v.original for v in self.views]), axis=0))]
        warped = [(v.position.id, v.image) for v in self.views] + [
            ('combined', self.image)]
        for ax, (title, img) in zip(axes.ravel(), originals + warped):
            ax.imshow(img)
            ax.axis('off')
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.set(title=title)
        fig.tight_layout()
        fig.canvas.draw()
        img_array = np.array(fig.canvas.renderer._renderer)
        plt.close('all')
        return img_array

    def bidirectional_similarity(self, flow_forward, flow_reverse):
        """

        :param flow_forward:
        :param flow_reverse:

        """
        height, width = flow_forward.shape[:2]
        grid = np.stack(np.mgrid[:height, :width], axis=2)

        projected = self.warp_flow(grid, flow_forward)
        reprojected = self.warp_flow(projected, flow_reverse)

        return ((grid - reprojected) ** 2).sum(axis=2)

    @staticmethod
    def warp_flow(image, flow):
        """

        :param image:
        :param flow:

        """
        height, width = image.shape[:2]
        grid = np.stack(np.mgrid[:height, :width], axis=2)

        if image.ndim == 2:
            image = image[..., np.newaxis]

        image = image.astype(np.float)
        coords = (grid - flow).transpose(2, 0, 1)

        return np.stack([skimage.transform.warp(channel, coords) for channel in
                         image.transpose(2, 0, 1)], axis=2)


class PostLabel(Label):
    def __init__(self, specimen_id, views, image):
        super(PostLabel, self).__init__(specimen_id, views)
        self._image = image

    def contrast(self):
        """
        Improves the contrast of the image (in-place).

        """
        self._image = improve_contrast(self._image)

    @property
    def image(self):
        return self._image
