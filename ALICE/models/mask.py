import numpy as np
import cv2
from typing import List
from operator import attrgetter



from alice.models.base import Base
from alice.predict import visualise_mask_predictions, mask_predictor


class Mask(Base):
    def __init__(self, mask: np.array, image):        
        super().__init__(image)
        self.mask = mask.astype(np.uint8)

    @property
    def y_midpoint(self):
        """
        Vertical midpoint of a mask
        """        
        return sum([min(np.where(self.mask == True)[0]), max(np.where(self.mask == True)[0])]) / 2 

    @property
    def contour(self):
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # If we have multiple contours, only use the largest one
        sorted_countors = sorted(contours, key=cv2.contourArea, reverse=True)
        return sorted_countors[0]
        
    def get_polygon(self, epsilon=None):
        """
        Get polygon around the mask

        Note: otherwise approx_best_fit_ngon takes many seconds so requires 
        low epsilon < 10 otherwise approx vertices are out 
        epsilon=5 works well
        """
        if not epsilon:
            epsilon = 0.02 * cv2.arcLength(self.contour, True)
            
        return cv2.approxPolyDP(self.contour, epsilon, True)

    def edges(self) -> np.array:
        """
        Get edges of a mask
        """
        # Diff will reduce width by 1, so prepend with extra column of 0s
        return np.diff(self.mask, prepend=np.zeros(self.image_height)[0])  

    def edge_points(self) -> List:   
        """
        Get the edges 
        """    
        # Find the indices of True values in the mask, and return row col (points)
        return [(col, row) for row, col in np.argwhere(self.edges())]  

    def _visualise(self, image):
        contour = self.contour
        image = cv2.drawContours(image, contour, -1, (255, 255, 0), 20)
        cv2.drawContours(image, [self.get_polygon()], -1, (0, 255, 0), 2)  # Draw in green        
        return image

    def subtract_mask(self, subtraction_mask):
        """
        Subtract mask from mask
        """
        self.mask = cv2.bitwise_and(self.mask, cv2.bitwise_not(subtraction_mask))        

class LabelMasks(Base):

    min_mask_size = 1500
    mask_model = mask_predictor

    def __init__(self, image):
        super().__init__(image)
        self.predictions = self.mask_model(self.image)
        self.label_masks = self._predictions_to_label_masks()
        self._clean_obscured_masks()
        
    def _predictions_to_label_masks(self):
        masks = self.predictions.get('instances').to("cpu").pred_masks.numpy()
        masks = self.filter_small_masks(masks)    
        label_masks = [Mask(m, self.image) for m in masks]
        # Sort by mask y midpoint
        label_masks.sort(key = attrgetter('y_midpoint'))        
        return label_masks

    def get_higher_labels_mask(self, label_index):
        """
        Get a mask representing all labels above the label_index
        """
        # Get masks higher up the label stack (higher masks have index 0)
        # CHECK MAKSKSSKSKSKSS. Which one needed this filter???
        higher_masks = self.label_masks[:label_index]
        combined_mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        for mask in higher_masks:
            cv2.fillPoly(combined_mask, [mask.get_polygon()], 255)    
        return combined_mask
    
    def _clean_obscured_masks(self):
        """
        Some detected areas are actually parts of masks higher in the stack
        So filter out obscured areas
        """
        for label_index, mask in enumerate(self.label_masks):
            higher_labels_mask = self.get_higher_labels_mask(label_index)
            mask.subtract_mask(higher_labels_mask)
            
    def get_masked_image(self, label_index):
        """
        Get the image with all higher labels masked in white
        """
        image = self.image.copy()
        higher_labels_mask = self.get_higher_labels_mask(label_index)
        kernel = np.ones((3, 3), np.uint8)
        # Lets dilate the mask a little to ensure it covers
        higher_labels_mask = cv2.dilate(higher_labels_mask, kernel, iterations=5)
        mask = (higher_labels_mask > 0).astype(np.uint8) 
        image[mask == 1] = [255, 255, 255]
        return image

    def filter_small_masks(self, masks: np.array):
        return [m for m in masks if np.count_nonzero(m) > self.min_mask_size]

    def __len__(self):
        return len(self.label_masks)

    def __getitem__(self, i):
        return self.label_masks[i]      
    
    def __iter__(self):
        yield from self.label_masks

    def _visualise(self, image):
        return visualise_mask_predictions(image, self.predictions)