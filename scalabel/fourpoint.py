import csv
import numpy as np
import skimage.draw
from skimage.transform import AffineTransform, estimate_transform, rescale, warp


def load_coordinates(path):
    points = np.zeros((4, 4, 2), dtype=np.float)

    with open(path, 'r+') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', skipinitialspace=True)
        header = next(reader)
        for camera, index, *coordinate in reader:
            points[int(camera), int(index)] = coordinate
        return points


class PerspectiveAlignment:
    def __init__(self, images, scale=0.25, coordinates='square.csv'):
        self.scale = scale
        self.images = [rescale(image, scale) for image in images]

        # points representing squares on the calibration grid
        self.point_list = load_coordinates(coordinates) * scale

        box_normal = np.array([[0, 100], [0, 0], [100, 0], [100, 100]])
        self.trans = [estimate_transform('projective', box_normal, points) for points in self.point_list]

    @property
    def registered_images(self):
        height, width = self.images[0].shape[:2]
        box_image = np.array([[0, height], [0, 0], [width, 0], [width, height]])
        bounds = [transformation.inverse(box_image) for transformation in self.trans]
        offset = np.stack(bounds).min(axis=0).min(axis=0)
        shape = np.stack(bounds).max(axis=0).max(axis=0) - offset
        scale = shape / np.array([4000, 4000])

        normalise = AffineTransform(scale=scale) + AffineTransform(translation=offset)

        return [warp(image, normalise + transformation, output_shape=(4000, 4000))[1000:-1000, 1000:-1000]
                for transformation, image in zip(self.trans, self.images)]

    @property
    def calibration_pattern_images(self):
        display_images = []
        for image, corners in zip(self.images, self.point_list):
            display_image = image.copy()
            display_image[skimage.draw.polygon_perimeter(corners[..., 1], corners[..., 0])] = 1
            display_images.append(display_image)
        return display_images
