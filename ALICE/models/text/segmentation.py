import numpy as np
import cv2
from scipy.stats import zscore
import scipy.cluster.hierarchy as hcluster
from collections import OrderedDict

from alice.craft import Craft
from alice.utils import random_colour, min_max
from alice.utils.cluster import ClusterVerticalInterval
from alice.models.base import Base
from alice.models.geometric import Rectangle, Point
from alice.models.text.line import TextLine
from alice.config import logger



class TextLineSegmentation(Base):

    craft = Craft()
    cluster = ClusterVerticalInterval()

    def __init__(self, image):
        prediction = self.craft.detect(image)        
        super().__init__(prediction['image'])
        self.heatmap = prediction['resized_heatmap']
        self.cluster_bboxes = self._cluster_bboxes()
        self.text_lines = self._get_text_lines()     

    def _cluster_bboxes(self):
        heatmap_rects = self._get_heatmap_bboxes()   
        if heatmap_rects.any():
            return self.cluster(heatmap_rects)
        else:
            logger.debug_image(self.heatmap, 'CRITICAL- No characters detected in heatmap')
            logger.warning("No characters detected in heatmap")
            return []
    
    def _get_text_lines(self):     

        text_lines = {}
        for i, cluster in enumerate(self.cluster_bboxes):
            if not cluster.is_valid(self.image_width): continue
            masked_image = self._get_masked_image(i, self.cluster_bboxes) 
            text_lines[i] = TextLine(cluster.bboxes, masked_image)
            
        return text_lines
                  
    def _get_masked_image(self, i, clusters):
        """
        Get image with other cluster's characters masked 
        """
        image = self.image.copy()
        mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        other_clusters = clusters[:i] + clusters[i+1:]
        
        for other_cluster in other_clusters:
            cv2.fillPoly(mask, [other_cluster.hull], color=1)
            
        # Dilate mask a little to cover the edges of the chars
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=3)
        # Mask areas of image with white
        image[dilated_mask == 1] = [255, 255, 255]         
        return image
                        
    def _mask_bboxes(self, binary_mask):
        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)                
        # Filter out smaller contours
        min_area = 4
        contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
        bboxes = []        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Remove any boxes along the top/bottom edges - overlapping from other rows
            if y == 0 or (y + h) >= self.image_height: continue            
            bboxes.append([(x, y), (x + w, y + h)])   
            
        return np.array(bboxes)         
    
    def _get_heatmap_bboxes(self):
        """
        Draw rectangles around points of heatmap density (representing letters)   
        """
        b, g, r = cv2.split(self.heatmap)
        threshold = 120
        # Use blue channel - 'hottest' so each letter will be separate
        binary_mask = b < threshold  
        return self._mask_bboxes(binary_mask)
               
    def _visualise(self, image):

        for cluster_bbox in self.cluster_bboxes.values():
            cluster_bbox.visualise(image)

        return image
    
    def visualise_heatmap(self):
        image = self.heatmap.copy()
        return self.visualise(image)
    
    def __len__(self):
        return len(self.text_lines)  
