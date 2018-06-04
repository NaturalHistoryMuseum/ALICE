import skimage.draw
from skimage.transform import rescale

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
        self.images = {camera_id: rescale(image, scale) for camera_id, image in
                       images}
        self.calibrator = calibrator
        self.calibrator.scale(scale)

    @property
    def corrected_images(self):
        """
        Applies transformations for each image and returns the warped images.
        :return: image objects
        """
        return [self.calibrator.views[c].apply_transform(i) for c, i in
                self.images.items() if c in self.calibrator.views.keys()]

    @property
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
            corners = self.calibrator.views[camera].coordinates
            display_image[
                skimage.draw.polygon_perimeter(corners[..., 1], corners[..., 0])] = 1
            display_images.append(display_image)
        return display_images
