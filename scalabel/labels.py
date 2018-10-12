import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import skimage.transform
from scalabel.pygco import cut_from_graph
from sklearn.neighbors import NearestNeighbors


class MultipleTransformations:
    def __init__(self, transform_type, num_angles):
        self.models = [transform_type() for _ in range(num_angles)]

    def estimate(self, points):
        for current_points, trans in zip(points.transpose(1, 0, 2)[:-1], self.models):
            trans.estimate(current_points, points[:, -1])
        return self

    def residual(self, points):
        total = []
        for current_points, trans in zip(points.transpose(1, 0, 2)[:-1], self.models):
            distance = ((trans(current_points) - points[:, -1]) ** 2).sum(axis=1)
            total.append(np.minimum(200000, np.ceil(distance)).astype(np.int32))
        return np.stack(total).max(axis=0)

    @classmethod
    def initialise_model(cls, points):
        num_points = points.shape[0]
        minimal_subset = points[np.random.choice(num_points, 8, replace=False)]
        return cls(skimage.transform.ProjectiveTransform,
                   num_angles=4).estimate(minimal_subset)


def visualise_labels(images, points, labels, models, show=True):
    img = np.concatenate(images, axis=1)
    plt.imshow(img)
    for m in models:
        x = (points[labels == m, :, 0] + np.arange(4)[np.newaxis, :] * images[0].shape[
            1]).flatten()
        y = points[labels == m, :, 1].flatten()
        plt.plot(x, y, '+', markersize=1)
    if show:
        plt.show()


def crop_to_points(points, border=0.5):
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


def separate_labels(images, points, visualise=False):
    best_models, labels = [0, 0]

    if visualise:
        visualise_labels(images, points, labels, [model for model, index in best_models],
                         show=False)
        plt.savefig('labels.png', dpi=1000, bbox_inches='tight')
        plt.close()

    label_images = []
    for model, index in best_models:
        crops = [crop_to_points(points[labels == index, k]) for k in range(len(images))]
        bounding_dimensions = [max(crop.stop - crop.start for crop in dim) for dim in
                               zip(*crops)]
        height, width = bounding_dimensions
        equal_crops = [
            (slice((y.stop + y.start - height) // 2, (y.stop + y.start + height) // 2),
             slice((x.stop + x.start - width) // 2, (x.stop + x.start + width) // 2)) for
            y, x in crops]
        current_label_image = [image[crop] for image, crop in zip(images, equal_crops)]
        if all([im.size > 0 for im in current_label_image]):
            label_images.append(current_label_image)

    return label_images


class LabelFinder(object):
    def __init__(self, images, features):
        self.images = images
        self.features = features
        self.best_models, self.labels = self.pearl()

    def pearl(self, K=500, max_iterations=30, minimum_support=10):
        eight_point = self.features.reshape(-1, 8)
        # for row in eight_point:
        #     r = row.reshape(-1, 2)
        #     fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20,20), sharex=True, sharey=True)
        #     for x, img, ax in zip(r, self.images, axes.ravel()):
        #         ax.imshow(img)
        #         ax.axis('off')
        #         ax.plot(x[...,0], x[...,1], 'r+')
        #     plt.show()
        unique_edges = set(self.edges_nearest(eight_point, num_neighbours=20))
        edges = np.stack(unique_edges).astype(np.int32)

        models = [MultipleTransformations.initialise_model(self.features) for _ in range(K)]

        pairwise = 1 - np.eye(len(models), dtype=np.int32)

        for iteration in range(max_iterations):
            unary = np.stack([model.residual(self.features) for model in models], axis=1)
            labels = cut_from_graph(edges, unary, pairwise)
            for i, model in enumerate(models):
                inliers = (labels == i)
                if inliers.sum() >= minimum_support:
                    model.estimate(self.features[inliers])
                else:
                    models[i] = MultipleTransformations.initialise_model(self.features)
            print('Completed iteration {}'.format(iteration))

        inliers = np.bincount(labels.flatten(), minlength=len(models))
        return [(model, i) for i, (model, support) in enumerate(zip(models, inliers)) if
                support >= minimum_support], labels

    @staticmethod
    def edges_nearest(points, num_neighbours):
        neighbors = NearestNeighbors(n_neighbors=num_neighbours)
        neighbors.fit(points)

        num_points = points.shape[0]

        for i in range(num_points):
            distances, indices = neighbors.kneighbors(points[[i]])
            for j, (distance, index) in enumerate(
                    zip(distances.flatten(), indices.flatten())):
                if index != i:
                    yield (*sorted((i, index)), int(distance))
