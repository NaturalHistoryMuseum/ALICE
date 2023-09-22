import numpy as np
import cv2
from typing import List


from alice.config import LOG_DIR
from alice.craft import Craft
from alice.utils import random_colour, min_max
from alice.models.geometric import Rectangle


class TextLine():
    def __init__(self, bboxes: List[Rectangle], image):
        self.orig_image = image.copy()
        self.height, self.width = self.orig_image.shape[:2]
        self.bboxes = bboxes
        self.baseline_centroids = self._get_baseline_centroids()
        self.image = self.deskew()

    def _get_baseline_centroids(self):
        # Calculate the baseline centroids of the heatmap (letter) rectangles
        return np.array([
            (np.mean(bbox[:, 0], dtype=np.int32), max(bbox[:, 1])) for bbox in self.bboxes
        ])   
        
    def get_line_best_fit(self):
        # Fit a polynomial of degree 2 (quadratic)
        degree = 2      
        x, y = self.baseline_centroids.T
        coefficients = np.polyfit(x, y, degree) 
        return np.poly1d(coefficients)

    def predict_y(self):
        poly = self.get_line_best_fit()
        x = np.linspace(0, self.width-1, self.width)        
        y_hat = poly(x)
        # Make sure predicted y is within the frame of the image 
        y_hat = np.clip(y_hat, 0, self.height-1)
        return x, y_hat

    def deskew(self):       
        y_offsets = self.y_offsets()
        deskewed = self.orig_image.copy()
        channels = deskewed.shape[2]
        # Roll down pixels by the offset from the line of best fit
        for i, y in enumerate(y_offsets):
            for c in range(channels):
                deskewed[:, i, c] = np.roll(deskewed[:, i, c], y)
            deskewed[:y, i, :] = 255

        # return deskewed
        return self.crop(deskewed)

    def crop(self, image):    
        x, y_hat = self.predict_y()    
        box_heights = np.diff(self.bboxes[:,:,1])
        max_height = np.max(box_heights).astype(int)                
        bottom_y = np.max(y_hat).astype(int)
        top_y = bottom_y - max_height

        # Add some padding to ensure all the letter forms are included
        # But without encroaching on other clusters
        top_padding, bottom_padding = self._get_padding(image, top_y, bottom_y)
        top_y -= top_padding
        bottom_y += bottom_padding       
                 
        cropped_image = image[top_y:bottom_y, :]           
        return cropped_image       
               
    def _get_padding(self, image, top_y, bottom_y):
        """
        Add padding but make sure it doesn't end up including other masked clusters
        And stays within the image canvas
        """
        padding = 10        
        
        # Add some padding, but make sure it's within the image canvas
        top_padding = min([top_y, padding])
        bottom_padding = min([self.height - bottom_y, padding])
        
        # Create binary mask to detect other masked clusters
        binary_mask = self._get_mask(image)

        adjusted_top_padding = self._get_top_padding(binary_mask, top_y, top_padding)
        adjusted_bottom_padding = self._get_bottom_padding(binary_mask, bottom_y, bottom_padding)

        return adjusted_top_padding, adjusted_bottom_padding
    
    def get_mask(self):
        return self._get_mask(self.image)
    
    def _get_mask(self, image):
        """
        Mask all white pixels above 254
        """
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary_mask = grey > 254   
        return binary_mask     
             
    def _get_top_padding(self, binary_mask, top_y, top_padding):
        top_padding_region = binary_mask[top_y - top_padding:top_y:, :]        
        y_mask = np.argwhere(top_padding_region)
        # If there's nothing in y_mask there are no mask regions in the padding zone
        if y_mask.any():
            max_y_mask = np.max(y_mask[:,0])
            top_padding -= max_y_mask
        return top_padding

    def _get_bottom_padding(self, binary_mask, bottom_y, bottom_padding):
        padding_region = binary_mask[bottom_y:bottom_y+bottom_padding, :]
        y_mask = np.argwhere(padding_region) 
        if y_mask.any():
            min_y_mask = np.min(y_mask[:,0])            
            return min_y_mask 
            
        return bottom_padding             
             
    def y_offsets(self):
        x, y_hat = self.predict_y()  
        # Offset is the predicted y values minus the maximum 
        y_offsets = max(y_hat) - y_hat
        y_offsets = np.round(y_offsets)
        return y_offsets.astype(int)

    def visualise(self):
        image = self.orig_image.copy()
        xs, y_hat = self.predict_y()  
        for x, y in zip(xs.astype(int),y_hat.astype(int)):            
            image[y, x] = [255, 0, 0]            
        
        return image