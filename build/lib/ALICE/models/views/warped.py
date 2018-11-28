from skimage.transform import rescale

from .base import View


class WarpedView(View):
    """
    A view of a specimen or area of the specimen where the image is transformed.
    :param view_position: a defined position in the current calibration from which
                          this photo was taken
    :param image: the image as a numpy array
    :param transform: a function taking a numpy array image as an argument and
                      returning a numpy array image
    :param original: the original image - preserved through view transformations where
                     self.image may be warped

    """
    scale = 0.25

    def __init__(self, view_position, image, original, transform=None):
        if transform is not None:
            image = transform(image)
        super(WarpedView, self).__init__(view_position, image, original)

    @classmethod
    def from_view(cls, view):
        """
        Create a new instance of this class from another instance of a View class or
        subclass. Uses the view position's transform function and scales the image.
        :param view: the view to transform

        """
        scaled_img = rescale(view.image, cls.scale, mode='constant',
                             anti_aliasing=True, multichannel=True)
        scaled_view_position = view.position.scale(cls.scale)
        return cls(scaled_view_position, scaled_img, view.original,
                   scaled_view_position.apply_transform)

    @property
    def display(self):
        """
        Returns a display image (the warped image).
        :return: an image as a numpy array

        """
        return self.image
