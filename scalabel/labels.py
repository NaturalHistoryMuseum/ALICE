import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import skimage.transform
from sklearn.neighbors import NearestNeighbors


from scalabel.pygco import cut_from_graph


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
            distance = ((trans(current_points) - points[:, -1])**2).sum(axis=1)
            total.append(np.minimum(200000, np.ceil(distance)).astype(np.int32))
        return np.stack(total).max(axis=0)


def visualise_labels(images, points, labels, models, show=True):
    img = np.concatenate(images, axis=1)
    plt.imshow(img)
    for m in models:
        x = (points[labels == m, :, 0] + np.arange(4)[np.newaxis, :] * images[0].shape[1]).flatten()
        y = points[labels == m, :, 1].flatten()
        plt.plot(x, y, '+', markersize=1)
    if show:
        plt.show()


def edges_nearest(points, num_neighbours):
    neighbors = NearestNeighbors(n_neighbors=num_neighbours)
    neighbors.fit(points)

    num_points = points.shape[0]

    for i in range(num_points):
        distances, indices = neighbors.kneighbors(points[[i]])
        for j, (distance, index) in enumerate(zip(distances.flatten(), indices.flatten())):
            if index != i:
                yield (*sorted((i, index)), int(distance))


def initialise_models(points, K):
    num_points = points.shape[0]
    for k in range(K):
        minimal_subset = points[np.random.choice(np.arange(num_points), 8)]
        yield MultipleTransformations(skimage.transform.ProjectiveTransform, num_angles=4).estimate(minimal_subset)


def minimum_support(labels, support_size=10):
    remaining_models = np.unique(labels)
    support = [(labels == m).sum() for m in remaining_models]

    return sorted([remaining_models[m] for m in np.where(np.array(support) > support_size)[0]])


def pearl(points, K=500, max_iterations=30):
    unique_edges = set(edges_nearest(points.reshape(-1, 8), num_neighbours=20))
    edges = np.stack(unique_edges).astype(np.int32)

    models = list(initialise_models(points, K))

    labels_old = None
    model_changed = [True] * len(models)
    unary = np.zeros((points.shape[0], len(models)))

    pairwise = -np.eye(len(models), dtype=np.int32)

    for iteration in range(max_iterations):
        unary = np.stack([model.residual(points) if has_changed else unary[:, j]
                          for j, (model, has_changed) in enumerate(zip(models, model_changed))], axis=1)
        labels = cut_from_graph(edges, unary, pairwise)
        for i, model in enumerate(models):
            inliers = (labels == i)
            if inliers.sum() >= 10:
                model.estimate(points[inliers])
                model_changed[i] = True
            else:
                model_changed[i] = False
        print('Completed iteration {}'.format(iteration))

        if labels_old is not None and (labels == labels_old).all():
            break
        else:
            labels_old = labels

    return minimum_support(labels, support_size=10), models, labels


def crop_to_points(points, border=0.5):
    centre = np.median(points, axis=0, keepdims=True)
    distance = ((points - centre)**2).sum(axis=1)

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
    best_models, models, labels = pearl(points)

    if visualise:
        visualise_labels(images, points, labels, best_models, show=False)
        plt.savefig('labels.png', dpi=1000, bbox_inches='tight')
        plt.close()

    label_images = []
    for j, m in enumerate(best_models):
        crops = [crop_to_points(points[labels == m, k]) for k in range(len(images))]
        bounding_dimensions = [max(crop.stop - crop.start for crop in dim) for dim in zip(*crops)]
        height, width = bounding_dimensions
        equal_crops = [(slice((y.stop + y.start - height) // 2, (y.stop + y.start + height) // 2),
                        slice((x.stop + x.start - width) // 2, (x.stop + x.start + width) // 2)) for y, x in crops]
        current_label_image = [image[crop] for image, crop in zip(images, equal_crops)]
        if all([im.size > 0 for im in current_label_image]):
            label_images.append(current_label_image)

    return label_images


if __name__ == '__main__':
    images = [skimage.io.imread('outlabel{}.png'.format(i)) for i in range(4)]
    points = np.load('points.npy')

    label_images = separate_labels(images, points, visualise=True)

    for i, current_label_images in enumerate(label_images):
        for j, image in enumerate(current_label_images):
            ax = plt.subplot(len(label_images), len(current_label_images), i * len(current_label_images) + 1 + j)
            ax.imshow(image)
            ax.axis('off')
    plt.savefig('image_labels.png', dpi=200, bbox_inches='tight')
    plt.close()
