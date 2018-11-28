from matplotlib import pyplot as plt

from ALICE.models.view_position import ViewPosition


class View(object):
    """
    An image of a specimen from a specified position.
    :param view_position: a defined position in the current calibration from which
                          this photo was taken
    :param image: the image as a numpy array
    :param original: the original image - preserved through view transformations where
                     self.image may be warped

    """

    def __init__(self, view_position: ViewPosition, image, original):
        self.position = view_position
        self._image = image
        self._original = original

    @property
    def image(self):
        """
        Purely for lazy-loading purposes.
        :return:
        """
        return self._image

    @property
    def original(self):
        """
        Also for lazy-loading.
        :return:
        """
        return self._original

    @classmethod
    def from_view(cls, view):
        """
        Create a new instance of this class from another instance of a View class or
        subclass.
        :param view: the view to transform

        """
        return cls(view.position, view.image, view.original)

    @classmethod
    def from_file(cls, view_position, imgfile):
        """
        Create a new instance of this class by loading an image directly from a file.
        :param view_position: a defined position in the current calibration from which
                              this photo was taken
        :param imgfile: path to the image file

        """
        img = plt.imread(imgfile)
        return cls(view_position, img, img)

    @property
    def display(self):
        """
        Returns a display image (the image with the view position marked on it).
        :return: an image as a numpy array

        """
        return self.position.display(self.image)