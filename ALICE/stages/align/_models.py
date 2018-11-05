import numpy as np
import skimage
import skimage.transform
import skimage.transform
import pyflow
from matplotlib import pyplot as plt
from skimage.feature import match_descriptors
from skimage.measure import ransac

from ALICE.models import Label
from ALICE.models.views import WarpedView
from ALICE.models.viewsets import FeatureComparer
from ._warping import SimilarAsPossible, bidirectional_similarity


class AlignedLabel(Label):
    """
    A collection of View objects representing a single label. The view images are
    warped to align with each other.
    :param specimen_id: the ID of the associated specimen
    :param views: a list of View objects (views of the label)

    """

    def __init__(self, specimen_id, views):
        super(AlignedLabel, self).__init__(specimen_id, views)
        self.comparer = FeatureComparer.ensure_minimum(specimen_id, views, 10)
        # align homography (using FeaturesView objects)
        self.views[1:] = [self._homography(v) for v in self.comparer.views[1:]]
        self.views[0] = WarpedView(self.views[0].position, self.views[0].image,
                                   self.views[0].original)
        self.views[1:] = [self._regularised_flow(v) for v in self.views[1:]]

    def _homography(self, view):
        """
        Match features between the view and the base view, generate transform
        parameters, and warp the view image to align with the base view. Uses the
        RANSAC (random sample consensus) method.
        :param view: the view to compare to the base view and warp as necessary

        """
        # matches = match_descriptors(view.descriptors,
        #                             self.views[0].descriptors, cross_check=True)
        matches = self.comparer.get_matches(self.views[0], view)
        src = view.keypoints[matches[:, 0], ::-1]
        dst = self.comparer.base_view.keypoints[matches[:, 1], ::-1]

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

        valid = (bidirectional_similarity(flow, flow_reverse) < 2).flatten()

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
