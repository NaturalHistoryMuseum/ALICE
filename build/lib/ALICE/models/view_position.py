import numpy as np
import re
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from skimage import color, filters, measure, segmentation
from skimage.filters import gaussian
from skimage.measure import regionprops
from skimage.transform import (AffineTransform, SimilarityTransform, estimate_transform,
                               rescale, warp)

from ALICE.utils.image import improve_contrast


class ViewPosition(object):
    """
    Defines a single camera angle or view of the specimen.
    :param view_id: an identifier for the view, e.g. ALICE3 or just 3
    :param coordinates: the four corners of a rectangle as viewed from this angle;
            specified in the order [bottom left, top left, top right, bottom
            right]
    :param scaled: the current scale of the view relative to the original size

    """

    def __init__(self, view_id, coordinates: np.array, scaled=1):
        self.id = view_id
        self.coordinates = coordinates
        self._transform = None
        self.scaled = scaled

    @classmethod
    def click(cls, pattern, filename):
        """
        Create a new ViewPosition instance by selecting the coordinates of a square on
        a calibration image.
        :param pattern: the calibration image
        :param filename: the name of the file (for extracting the identifier - the
                         file is not read)

        """
        alice_id = re.findall('(ALICE\d)', filename)
        alice_id = alice_id[0] if len(alice_id) > 0 else filename
        view_id = input('View ID [{0}]: '.format(alice_id))
        view_id = (view_id if view_id != '' else alice_id).replace(' ', '_')
        coords = []

        def _click(event):
            coords.append((int(event.xdata), int(event.ydata)))
            plt.close()

        def _move(event):
            if not event.inaxes:
                return
            lx.set_ydata(int(event.ydata))
            ly.set_xdata(int(event.xdata))
            plt.draw()

        for t in ['red', 'green', 'blue', 'purple']:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title(t)
            lx = ax.axhline(color=t, linewidth=1)
            ly = ax.axvline(color=t, linewidth=1)
            plt.imshow(pattern)
            fig.canvas.mpl_connect('motion_notify_event', _move)
            fig.canvas.mpl_connect('button_press_event', _click)
            plt.plot([c[0] for c in coords], [c[1] for c in coords], 'r+')
            plt.show()

        return cls(view_id, np.array(coords))

    @property
    def transform(self):
        """
        The transformation needed to warp the coordinates into a
        square.
        :return: a transform

        """
        if self._transform is None:
            square = np.array([[0, 1], [0, 0], [1, 0], [1, 1]])
            self._transform = estimate_transform('projective',
                                                 square,
                                                 self.coordinates)
        return self._transform

    def move_coords(self, img_dim):
        """
        Transform the coordinates to take up the maximum amount of space and move into
        the center.
        :param img_dim: the dimensions of the pattern images

        """
        # scale
        bbox_size = self.coordinates.max(axis=0) - self.coordinates.min(axis=0)
        scale = (img_dim / bbox_size).min()
        coords = SimilarityTransform(scale=scale)(self.coordinates)

        # translation (move to centre)
        actual_centre = coords.mean(axis=0)
        target_centre = img_dim / 2
        translation = target_centre - actual_centre

        # make the transform
        self.coordinates = SimilarityTransform(translation=translation)(coords)
        self._transform = None

    def scale(self, scale):
        """
        Return a new ViewPosition with the coordinates scaled to the specified factor.
        :param scale: the scale factor
        :return: ViewPosition

        """
        return ViewPosition(self.id, self.coordinates * scale, self.scaled * scale)

    def detect_regions(self, image):
        """
        Use thresholding and region detection to attempt to find a square
        containing the specimen.
        :param image: the image to process
        :returns: an image cropped to a square (using the original height or width)

        """
        grey = color.rgb2gray(image)
        h, w = grey.shape

        threshold_value = filters.threshold_otsu(grey)
        binary = grey < threshold_value
        labelled = measure.label(binary)
        regions = sorted(measure.regionprops(labelled), key=lambda x: -x.bbox_area)[:20]
        centroids = np.array([region.centroid for region in regions])
        avg_centroid = centroids.mean(axis=0)

        mid_h, mid_w = avg_centroid
        x1 = max(0, int(mid_w - (h / 2))) if w > h else 0
        x2 = min(w, x1 + h) if w > h else w + 1
        y1 = max(0, int(mid_h - (w / 2))) if w < h else 0
        y2 = min(h, y1 + w) if w < h else h + 1
        self._transform = None
        return image.copy()[y1:y2, x1:x2]

    def crop_to_labels(self, image, h=None, w=None):
        """
        Attempts to find labels in the image by looking for brighter/whiter areas,
        then crops down to that area.
        :param w: the original image width to scale down from
        :param h: the original image height to scale down from
        :param image: the image to crop
        :return: a cropped image

        """
        if h is None:
            h = image.shape[0]
        if w is None:
            w = image.shape[1]
        crop_size = np.array([h, w]) * 0.7
        img = improve_contrast(image.copy(), discard=5)
        img = gaussian(img, multichannel=True)

        labels = segmentation.slic(img, compactness=100, n_segments=400)
        regions = sorted(regionprops(labels),
                         key=lambda x: img[list(zip(*x.coords))].mean().tolist())[
                  -2:]
        centroid = np.array([r.centroid for r in regions]).mean(axis=0)
        crop_y, crop_x = np.array([centroid - (crop_size / 2),
                                   centroid + (crop_size / 2)]).T.astype(int)
        crop_x -= crop_x.min() if crop_x.min() < 0 else 0
        crop_x -= (crop_x.max() - image.shape[1]) if crop_x.max() > image.shape[1] else 0
        crop_y -= crop_y.min() if crop_y.min() < 0 else 0
        crop_y -= (crop_y.max() - image.shape[0]) if crop_y.max() > image.shape[0] else 0
        self._transform = None
        cropped = improve_contrast(image.copy()[slice(*crop_y), slice(*crop_x)], 1)
        return cropped

    def apply_transform(self, image):
        """
        Apply the view transformation to the given image, warping it to
        correct perspective distortion.
        :param image: the image to warp
        :returns: a warped image object

        """
        h, w = image.shape[:2]
        image = self.detect_regions(image)
        #image = self.crop_to_labels(image, h, w)
        #self.move_coords(np.array([image.shape[:2]]))
        height, width = image.shape[:2]
        box_image = np.array([[0, height], [0, 0], [width, 0], [width, height]])
        bounds = self.transform.inverse(box_image)
        offset = np.stack(bounds).min(axis=0).min(axis=0)
        shape = np.stack(bounds).max(axis=0).max(axis=0) - offset
        output_size = [4000, 4000]
        scale = shape / np.array(output_size)

        normalise = AffineTransform(scale=scale) + AffineTransform(translation=offset)
        warped = warp(image, normalise + self.transform, output_shape=tuple(output_size),
                      mode='constant')
        return warped

    def display(self, image):
        """
        Displays the calibration coordinates plotted on the pattern image,
        and the warped pattern image calculated from the transform. Useful for
        debugging.
        :param image: the image to use; does not have to be a calibration pattern

        """
        display_image = rescale(image.copy(), self.scaled,
                                anti_aliasing=True,
                                multichannel=True)

        fs = tuple([i / 200 for i in display_image.shape[:2]])
        fig, axes = plt.subplots(ncols=2, figsize=fs)

        # unwarped
        axes[0].imshow(display_image)
        axes[0].plot(*self.coordinates.mean(axis=0), 'bo')
        perimeter = Polygon(self.coordinates, linewidth=2, facecolor='none',
                            edgecolor='r')
        axes[0].add_patch(perimeter)

        # warped
        axes[1].imshow(self.apply_transform(display_image))

        for ax in axes.ravel():
            ax.axis('off')
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
        fig.tight_layout()
        fig.canvas.draw()
        img_array = np.array(fig.canvas.renderer._renderer)
        plt.close('all')
        return img_array
