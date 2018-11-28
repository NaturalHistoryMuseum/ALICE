from matplotlib import pyplot as plt

from .base import View


class LoadingView(View):
    def __init__(self, view_position, img_path):
        super(LoadingView, self).__init__(view_position, None, None)
        self._imgpath = img_path

    @property
    def image(self):
        if self._image is None:
            self._image = plt.imread(self._imgpath)
        return self._image

    @property
    def original(self):
        if self._original is None and self._image is None:
            self._original = plt.imread(self._imgpath)
        elif self._original is None:
            self._original = self._image
        return self._original
