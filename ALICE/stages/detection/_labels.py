import numpy as np
from pygco import cut_from_graph
from sklearn.neighbors import NearestNeighbors

from ALICE.models import MultipleTransformations, Specimen, View
from ALICE.models.logger import logger
from ALICE.models.viewsets import Label
from ._features import FeaturesSpecimen


class LabelSpecimen(Specimen):
    """
    A collection of View objects representing a single specimen, where the common
    features have been found.
    :param specimen_id: the ID of the associated specimen
    :param views: a list of View objects (views of the specimen)
    :param keypoints: the coordinates of the common features in the views

    """

    def __init__(self, specimen_id, views, keypoints):
        super(LabelSpecimen, self).__init__(specimen_id, views)
        self.keypoints = keypoints
        self.labels = self._find_labels()

    @classmethod
    def from_specimen(cls, specimen):
        """
        Create a new instance of this class from a FeaturesSpecimen object.
        :param specimen: the specimen to transform

        """
        assert isinstance(specimen, FeaturesSpecimen)
        return cls(specimen.id, specimen.views, specimen.comparer.global_matches)

    def _find_labels(self):
        """
        Extracts corresponding regions of each view that look like labels.
        :return: a list of Label objects

        """
        best_models, kp_ix = self.pearl()
        label_views = []
        for model, ix in best_models:
            crops = [self._crop(self.keypoints[kp_ix == ix, k]) for k in
                     range(len(self.views))]
            max_stop = [max(crop.stop + crop.start for crop in dim) for dim in
                        zip(*crops)]
            min_stop = [min(crop.stop + crop.start for crop in dim) for dim in
                        zip(*crops)]
            bounds = [[crop.stop - crop.start for crop in dim] for dim in
                      zip(*crops)]
            try:
                height, width = [
                    max(c for c in d if (c + maxs) // 2 < s and (mins - c) // 2 > 0) for
                    d, maxs, mins, s in
                    zip(bounds, max_stop, min_stop, self.views[0].image.shape[::-1][1:])]
            except ValueError:
                continue
            equal_crops = [
                (slice((y.stop + y.start - height) // 2,
                       (y.stop + y.start + height) // 2),
                 slice((x.stop + x.start - width) // 2,
                       (x.stop + x.start + width) // 2))
                for y, x in crops]
            views = [View(view.position, view.image[crop], view.image[crop]) for
                     view, crop in zip(self.views, equal_crops)]
            current_label_view = Label(self.id, views)
            if all([lv.image.size > 0 for lv in current_label_view.views]):
                label_views.append(current_label_view)
        if len(label_views) == 0:
            logger.error(f'did not find any labels in {len(self.keypoints)} keypoints!')
            raise Exception('No labels found.')
        else:
            logger.debug(f'found {len(label_views)} labels.')
        return label_views

    def pearl(self, k=5000, max_iterations=30, minimum_support=10):
        """
        Finds groups of keypoints that look like labels. PEaRL: Propose, Expand,
        and ReLearn.
        :param k:
        :param max_iterations:
        :param minimum_support:
        :return: a list of models and indices to keypoints

        """
        edges = np.stack(set(self.edges_nearest)).astype(np.int32)
        models = [MultipleTransformations.initialise_model(self.keypoints) for _ in
                  range(k)]
        pairwise = 1 - np.eye(len(models), dtype=np.int32)

        for iteration in range(max_iterations):
            unary = np.stack([model.residual(self.keypoints) for model in models],
                             axis=1)
            labels = cut_from_graph(edges=edges,
                                       edge_weights=np.array([1] * len(edges)),
                                       unary_cost=unary, pairwise_cost=pairwise)
            for i, model in enumerate(models):
                inliers = (labels == i)
                if inliers.sum() >= minimum_support:
                    model.estimate(self.keypoints[inliers])
                else:
                    models[i] = MultipleTransformations.initialise_model(self.keypoints)
            logger.debug('Completed iteration {}'.format(iteration))

        inliers = np.bincount(labels.flatten(), minlength=len(models))
        return [(model, i) for i, (model, support) in enumerate(zip(models, inliers)) if
                support >= minimum_support], labels

    @property
    def edges_nearest(self):
        """ """
        points = self.keypoints.reshape(-1, len(self.views) * 2)
        neighbors = NearestNeighbors(n_neighbors=20)
        neighbors.fit(points)
        for i in range(points.shape[0]):
            distances, indices = neighbors.kneighbors(points[[i]])
            for j, (distance, index) in enumerate(
                    zip(distances.flatten(), indices.flatten())):
                if index != i:
                    yield (*sorted((i, index)), int(distance))

    @staticmethod
    def _crop(points, border=0.5):
        """
        Find the crop parameters for the given set of points.

        :param points: a numpy array of keypoints
        :param border: space to leave around the cropped region
        :returns: the crop parameters (not a cropped image)

        """
        centre = np.median(points, axis=0, keepdims=True)
        distance = ((points - centre) ** 2).sum(axis=1)

        points = points[distance < 0.7 * distance.max()]

        left, top = points.min(axis=0)
        right, bottom = points.max(axis=0)

        shape = bottom - top, right - left

        left = int(left - border * shape[1])
        top = int(top - border * shape[0])
        right = int(right + border * shape[1])
        bottom = int(bottom + border * shape[0])

        return slice(top, bottom), slice(left, right)
