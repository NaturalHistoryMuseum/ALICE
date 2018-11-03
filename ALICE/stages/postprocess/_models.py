from ALICE.models import Label
from ALICE.utils.image import improve_contrast


class PostLabel(Label):
    def __init__(self, specimen_id, views, image):
        super(PostLabel, self).__init__(specimen_id, views)
        self._image = image

    def contrast(self):
        """
        Improves the contrast of the image (in-place).

        """
        self._image = improve_contrast(self._image)

    @property
    def image(self):
        return self._image
