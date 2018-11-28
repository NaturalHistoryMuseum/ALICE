import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from mrcnn import model as modellib, visualize

from ALICE.models.utils import LabelConfig
from .base import View


class RecognitionView(View):
    """
    A view where the image has been cropped to an area detected by an object recognition
    neural network model.
    :param view_position: a defined position in the current calibration from which
                          this photo was taken
    :param image: the image as a numpy array
    :param original: the original image - preserved through view transformations where
                     self.image may be warped or cropped

    """

    pad = 150

    def __init__(self, view_position, image, original):
        self.model_config = LabelConfig()
        with tf.device('/cpu:0'):
            self.model = modellib.MaskRCNN(mode='inference', model_dir='logs/',
                                           config=self.model_config)
        self.model.load_weights('ALICE/data/label_weights.h5', by_name=True)
        try:
            self.detected = self.model.detect([image])[0]
            boxes = self.detected['rois']
            y1, x1, y2, x2 = np.hstack(
                ((boxes.min(axis=0) - self.pad)[:2], (boxes.max(axis=0) + self.pad)[2:]))
            centroid = np.array([(y1 + y2) / 2, (x1 + x2) / 2])
            rad = max(abs(y2 - y1), abs(x2 - x1)) // 2
            y1, x1, y2, x2 = np.hstack((centroid - rad, centroid + rad)).astype(int)
            image = image[y1:y2, x1:x2]
        except IndexError or KeyError:
            self.detected = None
        super(RecognitionView, self).__init__(view_position, image, original)

    @property
    def display(self):
        """
        Returns a display image (the original image with detected regions marked).
        :return: an image as a numpy array

        """
        fs = tuple([i / 200 for i in self.original.shape[:2]])
        fig, ax = plt.subplots(figsize=fs)
        visualize.display_instances(self.original,
                                    self.detected['rois'],
                                    self.detected['masks'],
                                    self.detected['class_ids'],
                                    ['BG', 'label'],
                                    self.detected['scores'],
                                    ax=ax)
        ax.axis('off')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.autoscale(tight=True)
        fig.tight_layout()
        fig.canvas.draw()
        img_array = np.array(fig.canvas.renderer._renderer)
        plt.close('all')
        return img_array
