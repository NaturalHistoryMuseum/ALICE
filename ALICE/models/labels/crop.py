import numpy as np
import imutils
from typing import List
from collections import OrderedDict
from copy import deepcopy

from alice.config import logger
from alice.utils import min_max
from alice.models.labels.label import LabelValid, Label

class CropLabels:
    
    def __init__(self, labels: List[Label]):  
        self._labels = deepcopy(labels)

    @property
    def num_labels(self):
        return len(self._labels)
        
    def crop(self):             
        cropped_labels = self._crop(self._labels)
        rotated = self._rotate(cropped_labels)
        return rotated
 
    def _crop(self, labels):
        """
        Crop the the label
        """
        # Validate dimensions, removing outlisers
        quad_lengths = np.array([
            (label.quad.x_length, label.quad.y_length) for label in labels if label.is_valid()
        ])
        dimensions = np.array([min_max(d) for d in quad_lengths]) 
        if len(dimensions) > 1:
            labels, dimensions = self._validate_dimensions_shape(labels, dimensions)
        
        max_shortest_edge, max_longest_edge = np.max(dimensions[:,0]), np.max(dimensions[:,1]) 
        cropped = []
        for i, label in enumerate(labels):
            if label.is_valid():        
                cropped_label = label.crop(max_shortest_edge, max_longest_edge)    
                cropped.append((i, cropped_label))       
                
        # Use an ordered dict so order is retained for rotation  
        return OrderedDict(cropped)
    
    def _validate_dimensions_shape(self, labels, dimensions):    
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
                if labels[i].is_valid():
                    labels[i].set_valid(LabelValid.INVALID_SHAPE)

        # Mask the dimensions, excluding any invalid
        return labels, dimensions[outliers_mask]   

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
        min_axis, max_axis = dimensions.T
        max_outliers = self._get_outliers(max_axis)
        num_non_outliers = np.count_nonzero(max_outliers)
        # If we only have two or fewer, return
        if num_non_outliers <= 2:
            return max_outliers
            
        min_outliers = self._get_outliers(min_axis)
        # Combine both array of outliers using logical and into an outliers_mask
        return np.logical_and(min_outliers, max_outliers)
   
    
    def _get_outliers(self, data):

        # Calculate median deviation
        median = np.median(data)
        # calculate median absolute deviation
        deviation = np.sqrt((data - median)**2)
        max_deviation = np.max(data) / len(data)
        outliers = deviation < max_deviation

        if not any(outliers):
            lower_bound, upper_bound = min_max(data)
            # We have a wide range between upper & lower, so max deviation
            # outlier isn't going to work - instead slice data at 0.4 of the range          
            cut = lower_bound + (upper_bound - lower_bound) * 0.4
            outliers = data > cut            

        return outliers

    def _rotate(self, cropped_labels):
        first_landscape = self._get_first_landscape(cropped_labels)
        # Preserve order, but we don't care about actual label index positions 
        rotated = []
        for i in range(self.num_labels):
            if not i in cropped_labels: continue
            # If we don't have a first_landscape defined, just rotate all portrait images 90
            if first_landscape >= 0:
                rotation = (i - first_landscape) * 90
                # Adjust the rotation if it will make a landscape image not landscape
                if not self._is_valid_rotation(cropped_labels[i], rotation):
                    rotation -= 90
            elif not self._is_landscape(cropped_labels[i]):
                rotation = -90
            else:
                rotation = 0                    

            rotated_label = imutils.rotate_bound(cropped_labels[i], rotation)
            rotated.append(rotated_label)

        return rotated 
    
    def _is_valid_rotation(self, image, rotation):
        if self._is_landscape(image):
            return abs(rotation) in (0, 180)
        else:
            return abs(rotation) in (90, 270)
    
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