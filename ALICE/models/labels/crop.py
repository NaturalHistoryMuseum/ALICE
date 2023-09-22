import numpy as np
import imutils
from typing import List
from collections import OrderedDict

from alice.config import logger
from alice.utils import min_max
from alice.models.labels.label import LabelValid, Label

class CropLabels:
    
    def __init__(self, labels: List[Label]):  
        self._labels = labels

    @property
    def num_labels(self):
        return len(self._labels)
        
    def crop(self):             
        cropped_labels = self._crop()
        rotated = self._rotate(cropped_labels)
        return rotated
 
    def _crop(self):
        """
        Crop the the label
        This also checks that the label longest/shortest edge follows the same pattern
        If a box has an odd position, the nearest corner can shift between labels
        which then throws off the landscape alignment in _rotate() 
        """
        max_shortest_edge, max_longest_edge = self._validate_dimensions()        
        cropped = []
        for i, label in enumerate(self._labels):
            if label.is_valid():        
                cropped_label = label.crop(max_shortest_edge, max_longest_edge)    
                cropped.append((i, cropped_label))                            
        # Use an ordered dict so order is retained for rotation  
        return OrderedDict(cropped)
            
    def _rotate(self, cropped_labels):
        first_landscape = self._get_first_landscape(cropped_labels)
        # Preserve order, but we don't care about actual label index positions 
        rotated = []
        for i in range(self.num_labels):
            if not i in cropped_labels: continue
            # If we don't have a first_landscape defined, just rotate all portrait images 90
            if first_landscape >= 0:
                rotation = (i - first_landscape) * 90
            elif not self._is_landscape(cropped_labels[i]):
                rotation = -90
            else:
                rotation = 0

            rotated_label = imutils.rotate_bound(cropped_labels[i], rotation)
            rotated.append(rotated_label)

        return rotated 
    
    def _is_landscape(self, image): 
        h, w = image.shape[:2]
        return w > h
    
    def _get_first_landscape(self, cropped_labels): 
        # Capture the first landscape label - we'll use this as 
        # the base to rotate to 
        for i, cropped in cropped_labels.items():
            if self._is_landscape(cropped):
                return i
        # No landscape images detected
        return -1

    def _get_label_dimensions(self):
        dimensions = np.array([
            (label.quad.x_length, label.quad.y_length) for label in self._labels if label.is_valid()
        ])
        return np.array([min_max(d) for d in dimensions])   
    
    def _validate_dimensions(self):
        dimensions = self._get_label_dimensions()
        # We only have one label's dimensions - so cannot compare to validate
        if len(dimensions) > 1:
            dimensions = self._validate_dimensions_shape(dimensions)
            
        return np.max(dimensions[:,0]), np.max(dimensions[:,1])   
    
    def _validate_dimensions_shape(self, dimensions):    
        # Only 2 labels so cannot perform outlier detection
        if len(dimensions) == 2:
            outliers_mask = self._validate_shape_difference(dimensions)
        else:
            outliers_mask = self._validate_shape_deviation(dimensions)
            
        # Mark these labels as invalid so they won't be included in the merge
        for i in range(len(dimensions)):
            if not outliers_mask[i]:
                logger.info(f"Label {i} is not within normal range of other labels ")
                logger.info('Min Max: %s', dimensions)
                self._invalidate_label(i)    

        # Mask the dimensions, excluding any invalid
        return dimensions[outliers_mask]   
    
    def _invalidate_label(self, i):
        self._labels[i].set_valid(LabelValid.INVALID_SHAPE)

    def _validate_shape_difference(self, dimensions):
        """
        Validate shape lengths are not two disimilar from each other
        Used for validating shapes when we just have two labels 
        """

        # 20% of the mean of the width + length
        min_diff = np.mean(np.sum(dimensions, axis=0)) * 0.2
        # Calculate the maxium different along the edges
        diff = np.max(np.diff(dimensions, axis=0))
        if diff > min_diff:
            # Create a mask where True is set to the highest value
            max_value = np.max(dimensions[:, 1])
            mask = dimensions[:, 1] == max_value
        else:
            # Create an array of True True so we keep both
            mask = np.array([True] * 2)
        return mask
            
    def _validate_shape_deviation(self, dimensions):
        """
        Validate the deviations in shape, and remove outliers 
        """
        # Loop through the min max columns, checking they are within accepted deviations
        outliers = np.array([self._get_outliers(data) for data in dimensions.T])
        # Combine both array of outliers using logical and into an outliers_mask
        return np.logical_and(outliers[0], outliers[1])    
    
    def _get_outliers(self, data):
        # Calculate median deviation
        median = np.median(data)
        # calculate median absolute deviation
        deviation = np.sqrt((data - median)**2)
        max_deviation = np.max(data) / len(data)
        return deviation < max_deviation 