import matplotlib.pyplot as plt
import numpy as np
import re
from matplotlib.patches import Polygon
from skimage import color, filters, measure
from skimage.transform import AffineTransform, estimate_transform, rescale, warp, SimilarityTransform


class View(object):
    """
    Defines a single camera angle or view of the specimen.
    :param view_id: an identifier for the view, e.g. ALICE3 or just 3
    :param coordinates: the four corners of a rectangle as viewed from this angle;
                        specified in the order [bottom left, top left, top right,
                        bottom right]
    """

    def __init__(self, view_id, coordinates: np.array):
        self.id = view_id
        self.coordinates = coordinates
        self._transform = None
        self.scaled = 1
        self.pattern = None

    @classmethod
    def click(cls, pattern, filename):
        alice_id = re.findall('(ALICE\d)', filename)
        alice_id = alice_id[0] if len(alice_id) > 0 else filename
        view_id = input('View ID [{0}]: '.format(alice_id))
        view_id = (view_id if view_id != '' else alice_id).replace(' ', '_')
        coords = []

        def _click(event):
            coords.append((int(event.xdata), int(event.ydata)))
            plt.close()

        for t in ['red', 'green', 'blue', 'purple']:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title(t)
            plt.imshow(pattern)
            fig.canvas.mpl_connect('button_press_event', _click)
            plt.plot([c[0] for c in coords], [c[1] for c in coords], 'r+')
            plt.show()

        vw = View(view_id, np.array(coords))
        vw.pattern = pattern
        return vw

    @property
    def transform(self):
        """
        Recalculate the transformation needed to warp the coordinates into a square.
        :return:
        """
        if self._transform is None:
            square = np.array([[0, 1], [0, 0], [1, 0], [1, 1]])
            self._transform = estimate_transform('projective',
                                                 square,
                                                 self.coordinates)
        return self._transform

    def move_coords(self, image):
        img_dim = np.roll(image.shape[:2], 1)

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
        Scale the coordinates and recalculate the transformation.
        :param scale: the scale factor
        """
        self.coordinates *= scale
        self.scaled *= scale
        self._transform = None

    def detect_regions(self, image):
        """
        Use thresholding and region detection to attempt to find a square containing
        the specimen.
        :param image: the image to process
        :return: an image cropped to a square (using the original height or width)
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

    def apply_transform(self, image):
        """
        Apply the view transformation to the given image, warping it to correct
        perspective distortion.
        :param image: the image to warp
        :return: a warped image object
        """
        #image = self.detect_regions(image)
        self.move_coords(image)
        self.pattern = image
        height, width = image.shape[:2]
        box_image = np.array([[0, height], [0, 0], [width, 0], [width, height]])
        bounds = self.transform.inverse(box_image)
        offset = np.stack(bounds).min(axis=0).min(axis=0)
        shape = np.stack(bounds).max(axis=0).max(axis=0) - offset
        output_size = [4000, 4000]
        scale = shape / np.array(output_size)

        normalise = AffineTransform(scale=scale) + AffineTransform(translation=offset)
        warped_coords = (normalise + self.transform).inverse(self.coordinates)
        midpoint = warped_coords.mean(axis=0).astype(int)
        min_side = midpoint - 1000
        max_side = midpoint + 1000
        min_side[min_side < 0] = 0
        max_side[max_side > output_size[0]] = output_size[0]
        warped = warp(image, normalise + self.transform, output_shape=tuple(output_size),
                      mode='constant')
        cropped = warped[min_side[0]:max_side[0], min_side[1]:max_side[1]]
        return warped

    def display(self, pattern_image=None, set_image=True):
        """
        Displays the calibration coordinates plotted on the pattern image, and the
        warped pattern image calculated from the transform. Useful for debugging.
        :param pattern_image: the image to use; does not have to be a calibration pattern
        :param set_image: if True, will set the image as the view's calibration pattern
        """
        if pattern_image is None and self.pattern is None:
            return
        elif pattern_image is None:
            display_image = self.pattern
        else:
            display_image = rescale(pattern_image.copy(), self.scaled,
                                    anti_aliasing=True,
                                    multichannel=True)
            if set_image:
                self.pattern = display_image

        fs = tuple([i / 200 for i in display_image.shape[:2]])
        fig, axes = plt.subplots(ncols=2, figsize=fs)

        # unwarped
        axes[0].imshow(display_image)
        axes[0].plot(*self.coordinates.mean(axis=0), 'bo')
        axes[0].axis('off')
        perimeter = Polygon(self.coordinates, linewidth=3, facecolor='none',
                            edgecolor='r')
        axes[0].add_patch(perimeter)

        # warped
        axes[1].imshow(self.apply_transform(display_image))
        axes[1].axis('off')

        plt.show()
