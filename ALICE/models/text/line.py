import numpy as np
import cv2


from alice.config import LOG_DIR
from alice.craft import Craft
from alice.utils import random_colour, min_max
from alice.models.geometric import Rectangle


class TextLine():
    def __init__(self, rect: Rectangle, image, baseline_centroids):
        self.rect = rect
        x1, y1, x2, y2 = rect.as_numpy().ravel()
        self.orig_image = image[y1:y2, x1:x2]
        self.height, self.width = self.orig_image.shape[:2]
        # Baselines are calculate to the whole image - adjust
        baseline_centroids[:, 1] -= y1
        self.baseline_centroids = baseline_centroids
        self.image = self.deskew()
             
    def get_line_best_fit(self):
        # Fit a polynomial of degree 2 (quadratic)
        degree = 2      
        x, y = self.baseline_centroids.T
        coefficients = np.polyfit(x, y, degree) 
        return np.poly1d(coefficients)

    def deskew(self):
        poly = self.get_line_best_fit()
        x = np.linspace(0, self.width-1, self.width)        
        y_hat = poly(x)   
        # Offset is the predicted y values minus the maximum  
        y_offsets = max(y_hat) - y_hat
        deskewed = self.orig_image.copy()
        channels = deskewed.shape[2]
        # Roll down pixels by the offset from the line of best fit
        for i, y_offset in enumerate(y_offsets):
            y = round(y_offset)
            for c in range(channels):
                deskewed[:, i, c] = np.roll(deskewed[:, i, c], y)
            deskewed[:y, i, :] = 255
        return deskewed

    def visualise(self):
        image = self.orig_image.copy()
        poly = self.get_line_best_fit()
        xs = np.linspace(0, self.width -1, self.width)
        y_hat = poly(xs)
        for x, y in zip(xs.astype(int),y_hat.astype(int)):
            image[y, x] = [255, 0, 0]            
        
        return image