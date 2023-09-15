import numpy as np
import imutils
from scipy.stats import mode

from alice.config import logger
from alice.utils import min_max
from alice.models.text import TextDetection
from alice.models.labels import LabelValid

class LabelQuartet:
    """
    The four labels
    """
    def __init__(self):
        self._labels = []
        
    def add_label(self, label):
        self._labels.append(label)

    def get_cropped_labels(self):
        max_shortest_edge, max_longest_edge = self.get_dimensions()
        first_landscape = 0
        cropped = {}
        for i, label in enumerate(self._labels):
            if not label.is_valid(): continue

            cropped_image = label.crop(max_shortest_edge, max_longest_edge)
            h, w = cropped_image.shape[:2]
            is_landscape = w > h
            # Capture the first landscape label - we'll use this as 
            # the base to rotate to            
            if not first_landscape and is_landscape:
                first_landscape = i

            cropped[i] = label.crop(max_shortest_edge, max_longest_edge)

        # Rotate all images to the first landscape one
        rotated = {}
        for i in range(len(self._labels)):
            if not i in cropped: continue
            rotation = (i - first_landscape) * 90
            rotated[i] = imutils.rotate_bound(cropped[i], rotation)

        return rotated

    def count_valid(self):
        return len(list(self.iter_valid()))
    
    def iter_valid(self):
        for label in self._labels:
            if label.is_valid():
                yield label

    def _get_dimensions_min_max(self):
        dimensions = np.array([
            (label.quad.x_length, label.quad.y_length) for label in self.iter_valid()
        ])
        return np.array([min_max(d) for d in dimensions]) 
        
    def get_dimensions(self):
        """
        Get edge dimensions for all labels in the view
        """
        min_maxes = self._get_dimensions_min_max()
        min_maxes = self.validate_shape_homogeneity(min_maxes)
        return np.max(min_maxes[:,0]), np.max(min_maxes[:,1])        

    def validate_shape_homogeneity(self, min_maxes):

        if len(min_maxes) <= 1:
            return min_maxes
            
        if len(min_maxes) == 2:
            outliers_mask = self._validate_shape_difference(min_maxes)
        else:
            outliers_mask = self._validate_shape_deviation(min_maxes)
            
        # Mark these labels as invalid so they won't be included in the merge
        for i, label in enumerate(self.iter_valid()):
            if not outliers_mask[i]:
                logger.info(f"Label {i} is not within normal range of other labels ")
                logger.info('Min Max: %s', min_maxes)     
                logger.debug_image(label.visualise(), f'label-{i}-shape-outlier')                        
                label.set_valid(LabelValid.INVALID_SHAPE)
            
        # Mask the min maxes value, so outlines won't be used in dimension calculations 
        return min_maxes[outliers_mask]              

    def _validate_shape_difference(self, min_maxes):
        """
        Validate shape lengths are not two disimilar from each other
        Used for validating shapes when we just have two labels 
        """
        min_diff = np.mean(min_maxes) / 4
        # Calculate the maxium different along the edges
        diff = np.max(np.diff(min_maxes, axis=1))
        if diff > min_diff:
            # Create a mask where True is set to the highest value
            max_value = np.max(min_maxes[:, 1])
            mask = min_maxes[:, 1] == max_value
        else:
            # Create an array of True True so we keep both
            mask = np.array([True] * 2)
        return mask
            
    def _validate_shape_deviation(self, min_maxes):
        """
        Validate the deviations in shape, and remove outliers 
        """
        # Loop through the min max columns, checking they are within accepted deviations
        outliers = np.array([self._get_outliers(data) for data in min_maxes.T])
        # Combine both array of outliers using logical and into an outliers_mask
        return np.logical_and(outliers[0], outliers[1])    
    
    def _get_outliers(self, data):
        # Calculate median deviation
        median = np.median(data)
        # calculate median absolute deviation
        deviation = np.sqrt((data - median)**2)
        max_deviation = np.mean(data) / len(data)
        # FIXME: Why is one of the labels here being disallowed??
        # print(max_deviation)
        # print(deviation)
        # print(deviation < max_deviation)
        # re sub for corner label
        return deviation < max_deviation   
    
    def get_labels(self):
        # Return named tuples - orig, warped
        cropped_labels = self.get_cropped_labels()          
        labels_text = []
        for cropped_label in cropped_labels.values():
            labels_text.append(TextDetection(cropped_label))
            
        modal = mode([len(label_text) for label_text in labels_text]) 
        self._validate_num_textlines_per_label(labels_text, modal.mode) 
                    
    def _validate_num_textlines_per_label(labels_text, mode):
        for label_text in labels_text:
            if len(label_text) != mode:
                logger.info("Number of text lines %s does not match mode %s - recalculating clusters", len(label_text), mode)
                label_text.recalculate_clusters(mode)
            
            
            
        