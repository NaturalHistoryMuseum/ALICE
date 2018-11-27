import numpy as np
from matplotlib import pyplot as plt

from ALICE.models.views import LoadingView
from .base import ViewSet


class Specimen(ViewSet):
    """
    A collection of View objects representing a single specimen.
    :param specimen_id: the ID of the associated specimen
    :param views: a list of View objects (views of the specimen)

    """

    def __init__(self, specimen_id, views):
        super(Specimen, self).__init__(views)
        self.id = specimen_id
        self.labels = []

    @classmethod
    def from_images(cls, specimen_id, image_files):
        """
        Create a new specimen instance by loading images from files to create views.
        :param specimen_id: the ID of the specimen
        :param image_files: paths to the images of this specimen
        :return: Specimen

        """
        views = [LoadingView(v, img) for v, img in image_files]
        return cls(specimen_id, views)

    @classmethod
    def from_specimen(cls, specimen):
        """
        Create a new instance of this class from another instance of a Specimen class or
        subclass.
        :param specimen: the specimen to transform

        """
        return cls(specimen.id, specimen.views)

    @property
    def display(self):
        """
        Returns a display image (the original view images, then one row for each label).
        :return: an image as a numpy array

        """
        nrow = len(self.labels) + 1
        ncol = len(self.views)
        rows = [(self.views[0].image, len(self.views))]
        viewfig, viewaxes = plt.subplots(ncols=ncol, figsize=self._figsize(rows))
        for ax, view in zip(viewaxes, self.views):
            ax.imshow(view.image if len(self.labels) == 0 else view.original)
            ax.axis('off')
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.set(title=view.position.id)
        viewfig.tight_layout()
        viewfig.canvas.draw()
        views = np.array(viewfig.canvas.renderer._renderer)

        rows = [(views, 1),
                *[(l.display, 1) for l in self.labels]]
        fig, axes = plt.subplots(nrows=nrow,
                                 figsize=self._figsize(rows))
        if nrow == 1:
            axes = [axes]
        for ax, img in zip(axes, [views] + [l.display for l in self.labels]):
            ax.imshow(img)
            ax.axis('off')
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
        fig.tight_layout()
        fig.canvas.draw()
        img_array = np.array(fig.canvas.renderer._renderer)
        plt.close('all')
        return img_array
