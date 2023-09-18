import numpy as np
import cv2
from scipy.stats import zscore
import scipy.cluster.hierarchy as hcluster


from alice.craft import Craft
from alice.utils import random_colour, min_max
from alice.utils.cluster import ClusterVerticalInterval
from alice.models.base import Base
from alice.models.geometric import Rectangle, Point
from alice.models.text.line import TextLine



class TextLineSegmentation(Base):

    craft = Craft()
    cluster = ClusterVerticalInterval()

    def __init__(self, image):
        prediction = self.craft.detect(image)        
        super().__init__(prediction['image'])
        self.heatmap = prediction['resized_heatmap']
        self.text_lines = self._get_text_lines()

    def _get_text_lines(self):
        text_lines = []
        for cluster_bbox, masked_image, baseline_centroids in self._iter_clusters():
            text_lines.append(TextLine(cluster_bbox, masked_image, baseline_centroids))
        return text_lines

    def _iter_clusters(self):
        
        line_bboxes = self._get_heatmap_line_bboxes()
        line_rects = [Rectangle.from_numpy(bbox) for bbox in line_bboxes]  
        clusters = self._get_character_bbox_clusters()

        for i, cluster in enumerate(clusters):
            cluster_bbox = self._get_cluster_bbox(cluster, line_rects)
            masked_image = self._get_masked_image(i, clusters)  
            baseline_centroids = self._get_baseline_centroids(cluster)
            yield cluster_bbox, masked_image, baseline_centroids
            
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
                        
    @staticmethod
    def _get_baseline_centroids(cluster):
        # Calculate the baseline centroids of the heatmap (letter) rectangles
        return np.array([
            (np.mean(bbox[:, 0], dtype=np.int32), max(bbox[:, 1])) for bbox in cluster.bboxes
        ])         

    def _get_cluster_bbox(self, cluster, line_rects):
        """
        Get a bounding box around a cluster
        """
        cluster_bbox = cluster.get_bounding_box()
        padded_ymin, padded_ymax = self._bbox_get_padding(cluster_bbox, line_rects)
        return Rectangle(Point(0, padded_ymin), Point(self.image_width, padded_ymax))        
        
    def _bbox_get_padding(self, cluster_bbox, line_rects):  
        """
        Loop through the line rects, and extend the bbox min/max y coords
        line rects have been created using a more permissive mask, so span the whole
        row, and cluster bbox will be expanded to the new mask, if intersect ratio of more than 0.75
        
        """
        yy = cluster_bbox.yy

        # Add some default padding
        miny, maxy = min_max(np.array(yy))
        if miny > 2: yy.append(miny-2)
        if maxy < self.image_height-2: yy.append(maxy+2)
                
        for line_rect in line_rects:                
            intersection = cluster_bbox.intersection(line_rect)
            r = max([intersection.area / line_rect.area, intersection.area / cluster_bbox.area])
            if r > 0.75:
                # Only extend if new cluster height is line height is within .3 of the original
                if line_rect.height < (cluster_bbox.height * 1.3):
                    yy.extend(line_rect.yy)
        
        return min_max(np.array(yy))

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
    
    def _get_heatmap_line_bboxes(self):
        b, g, r = cv2.split(self.heatmap)
        threshold = 120
        # Use green channel - 'cooler' - more permissive with characters merged
        binary_mask = g > threshold  
        return self._mask_bboxes(binary_mask)
        
    def _get_heatmap_character_bboxes(self):
        """
        Draw rectangles around points of heatmap density (representing letters)   
        """
        b, g, r = cv2.split(self.heatmap)
        threshold = 120
        # Use blue channel - 'hottest' so each letter will be separate
        binary_mask = b < threshold  
        return self._mask_bboxes(binary_mask)
    
    def _get_character_bbox_clusters(self):
        heatmap_rects = self._get_heatmap_character_bboxes()        
        return self.cluster(np.array(heatmap_rects))
               
    def _visualise(self, image):

        image = self.image.copy()
        clusters = self._get_character_bbox_clusters()
        for cluster in clusters:
            color = random_colour()
            for rect in cluster:
                pt1, pt2 = rect
                cv2.rectangle(image, pt1, pt2, color, 1)            
                
            baseline_centroids = self._get_baseline_centroids(cluster)
            for point in baseline_centroids:
                cv2.circle(image, point, 2, (255, 255, 0), -1)
                    
        return image
    
    def visualise_heatmap(self):
        image = self.heatmap.copy()
        return self._visualise(image)
    
    def __len__(self):
        return len(self.text_lines)   
