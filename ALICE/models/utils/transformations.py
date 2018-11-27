import numpy as np
import skimage.transform


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