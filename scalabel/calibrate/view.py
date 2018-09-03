import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
import re
from skimage.transform import AffineTransform, estimate_transform, rescale, warp


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
        self.transform = self._get_transform()
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

    def _get_transform(self):
        """
        Recalculate the transformation needed to warp the coordinates into a square.
        :return:
        """
        square = np.array([[0, 1], [0, 0], [1, 0], [1, 1]])
        return estimate_transform('projective',
                                  square,
                                  self.coordinates)

    def scale(self, scale):
        """
        Scale the coordinates and recalculate the transformation.
        :param scale: the scale factor
        """
        self.coordinates *= scale
        self.transform = self._get_transform()
        self.scaled *= scale

    def apply_transform(self, image):
        """
        Apply the view transformation to the given image, warping it to correct
        perspective distortion.
        :param image: the image to warp
        :return: a warped image object
        """
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
        return warp(image, normalise + self.transform, output_shape=tuple(output_size),
                    mode='constant')[min_side[0]:max_side[0], min_side[1]:max_side[1]]

    def display(self, pattern_image, set_image=True):
        """
        Displays the calibration coordinates plotted on the pattern image, and the
        warped pattern image calculated from the transform. Useful for debugging.
        :param pattern_image: the image to use; does not have to be a calibration pattern
        :param set_image: if True, will set the image as the view's calibration pattern
        """
        display_image = rescale(pattern_image.copy(), self.scaled)
        if set_image:
            self.pattern = display_image
        fs = tuple([i / 200 for i in display_image.shape[:2]])
        fig, axes = plt.subplots(ncols=2, figsize=fs)

        # unwarped
        axes[0].imshow(display_image)
        axes[0].axis('off')
        perimeter = Polygon(self.coordinates, linewidth=3, facecolor='none',
                            edgecolor='r')
        axes[0].add_patch(perimeter)

        # warped
        axes[1].imshow(self.apply_transform(display_image))
        axes[1].axis('off')

        plt.show()
