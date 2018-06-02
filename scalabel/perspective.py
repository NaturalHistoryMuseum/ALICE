import numpy as np
import skimage.draw
from skimage.transform import AffineTransform, rescale, warp

from scalabel.calibrate import Calibrator


class PerspectiveCorrection:
    def __init__(self, images, scale=0.25, calibrator=Calibrator.from_csv('square.csv')):
        self.scale = scale
        self.images = [rescale(image, scale) for image in images]
        self.calibrator = calibrator
        self.calibrator.scale(scale)

    @property
    def registered_images(self):
        height, width = self.images[0].shape[:2]
        box_image = np.array([[0, height], [0, 0], [width, 0], [width, height]])
        bounds = [transformation.inverse(box_image) for transformation in
                  self.calibrator.transforms]
        offset = np.stack(bounds).min(axis=0).min(axis=0)
        shape = np.stack(bounds).max(axis=0).max(axis=0) - offset
        scale = shape / np.array([4000, 4000])

        normalise = AffineTransform(scale=scale) + AffineTransform(translation=offset)

        return [warp(image, normalise + transformation, output_shape=(4000, 4000))[
                1000:-1000, 1000:-1000]
                for transformation, image in zip(self.calibrator.transforms, self.images)]

    @property
    def calibration_pattern_images(self):
        display_images = []
        for image, corners in zip(self.images, self.calibrator.coordinates):
            display_image = image.copy()
            display_image[
                skimage.draw.polygon_perimeter(corners[..., 1], corners[..., 0])] = 1
            display_images.append(display_image)
        return display_images
