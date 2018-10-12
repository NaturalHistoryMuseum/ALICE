import matplotlib.pyplot as plt
import numpy as np
import skimage.draw
from skimage.transform import rescale
from cached_property import cached_property

from scalabel.calibrate import Calibrator


class PerspectiveCorrection:
    def __init__(self, images, scale=0.25, calibrator=None):
        """
        Applies transformations on images to account for perspective distortion.
        :param images: a list of (camera_id, image_object) tuples; camera id should
                       match that in the calibrator
        :param scale: a float for rescaling the images and coordinates
        :param calibrator: a Calibrator object loaded with Views for the images; if
                           None, will attempt to load from 'square.csv'
        """
        if calibrator is None:
            calibrator = Calibrator.from_csv('square.csv')
        self.scale = scale
        self.images = {camera_id: rescale(image, scale, mode='constant',
                                          anti_aliasing=True, multichannel=True) for
                       camera_id, image in images}
        self.calibrator = calibrator
        self.calibrator.scale(scale)

    @cached_property
    def corrected_images(self):
        """
        Applies transformations for each image and returns the warped images.
        :return: image objects
        """
        return [self.calibrator.views[c].apply_transform(i) for c, i in
                self.images.items() if c in self.calibrator.views.keys()]

    @cached_property
    def calibration_pattern_images(self):
        """
        Shows the location of the calibration squares on the unwarped images.
        :return: unwarped image objects with a marked rectangle
        """
        display_images = []
        for camera, image in self.images.items():
            if camera not in self.calibrator.views.keys():
                continue
            display_image = image.copy()
            h, w, _ = display_image.shape
            corners = self.calibrator.views[camera].coordinates
            corners[..., 1] = np.clip(corners[..., 1], 0, h - 1)
            corners[..., 0] = np.clip(corners[..., 0], 0, w - 1)
            display_image[
                skimage.draw.polygon_perimeter(corners[..., 1], corners[..., 0])] = 1
            display_images.append(display_image)
        return display_images

    def visualise(self):
        warped = self.corrected_images
        display = self.calibration_pattern_images

        fig, axes = plt.subplots(4, 2, figsize=(12, 24), squeeze=True)
        for ax, view, (aligned, original) in zip(axes, self.calibrator.views,
                                                 zip(warped, display)):
            ax[0].imshow(original)
            ax[0].axis('off')
            ax[0].set(title=view)
            ax[1].imshow(aligned)
            ax[1].axis('off')

        plt.show()
