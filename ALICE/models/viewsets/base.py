import numpy as np
from matplotlib import pyplot as plt


class ViewSet(object):
    """
    A collection of View objects.
    :param views: a list of View objects
    """

    def __init__(self, views):
        self.views = views

    @staticmethod
    def _figsize(rows):
        """
        Helper method to get a figure size for display images.
        :param rows: a list of tuples of (example_image, number_in_row)
        :return: (w,h)
        """
        max_width = max(i.shape[1] * n for i, n in rows)
        height = sum([i.shape[0] for i, n in rows])
        dpi = plt.rcParams['figure.dpi']
        if max_width / dpi > 100:
            ratio = height / max_width
            w = 24
            h = int(w * ratio)
        else:
            w = max_width / dpi
            h = height / dpi
        return w, h

    @property
    def display(self):
        """
        Returns a display image (all view images in a row, with titles).
        :return: an image as a numpy array

        """
        fig, axes = plt.subplots(1, len(self.views),
                                 figsize=self._figsize(
                                     [(self.views[0].image, len(self.views))]),
                                 squeeze=True)
        for ax, view in zip(axes.ravel(), self.views):
            ax.imshow(view.image)
            ax.axis('off')
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.set(title=view.position.id)
        fig.tight_layout()
        fig.canvas.draw()
        img_array = np.array(fig.canvas.renderer._renderer)
        plt.close('all')
        return img_array

    def show(self):
        """
        Show the display image. Just a helper method for debugging.

        """
        plt.imshow(self.display)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def save(self, fn):
        """
        Save the display image as a file.
        :param fn: the file name/path

        """
        plt.imsave(fn, self.display)
