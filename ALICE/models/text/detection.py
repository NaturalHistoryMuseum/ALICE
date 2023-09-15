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

class TextDetection(Base):

    craft = Craft()
    cluster = ClusterVerticalInterval()

    def __init__(self, image):
        print(image.shape)
        prediction = self.craft.detect(image)        
        super().__init__(prediction['image'])
        self.heatmap = prediction['resized_heatmap']
        self.text_lines = self._get_text_lines()

    def _get_text_lines(self, num_clusters=None):
        line_bboxes = self._get_heatmap_line_bboxes()
        line_rects = [Rectangle.from_numpy(bbox) for bbox in line_bboxes]   
        
             
        clusters = self._get_clustered_character_bboxes(num_clusters)
        
        
        lines = []
        for cluster in clusters:
            bbox = self._get_cluster_bbox(cluster, line_rects)
            baseline_centroids = self._get_baseline_centroids(cluster)
            line = TextLine(bbox, self.image, baseline_centroids)
            lines.append(line)    
        return lines
    
    def recalculate_clusters(self, num_clusters):
        self.text_lines = self._get_text_lines(num_clusters)

    @staticmethod
    def _get_baseline_centroids(cluster):
        # Calculate the baseline centroids of the heatmap (letter) rectangles
        return np.array([
            (np.mean(rect[:, 0], dtype=np.int32), max(rect[:, 1])) for rect in cluster
        ])         

    def _get_cluster_bbox(self, cluster, line_rects):
        """
        Get a bounding box around a cluster
        """
        ymin, ymax = min_max(cluster[:,:,1])
        xmin, xmax = min_max(cluster[:,:,0])
        cluster_bbox = Rectangle(Point(xmin, ymin), Point(xmax, ymax))
        padded_ymin, padded_ymax = self._bbox_get_padding(cluster_bbox, line_rects)
        return Rectangle(Point(0, padded_ymin), Point(self.image_width, padded_ymax))        
        
    @staticmethod
    def _bbox_get_padding(bbox, line_rects):  
        """
        Loop through the line rects, and extend the bbox min/max y coords
        line rects have been created using a more permissive mask, so this
        is a smarter way of adding padding to the bbox
        
        """
        yy = bbox.yy
        for line_rect in line_rects:    
            intersection = bbox.intersection(line_rect)
            r = max([intersection.area / line_rect.area, intersection.area / bbox.area])
            if r > 0.75:
                yy.extend(line_rect.yy)

        return min_max(np.array(yy))
        
    @staticmethod
    def _filter_outliers(bboxes):
        """
        Filter any outlier bboxes by height (joined up letters)
        """
        threshold = 3
        bboxes = np.array(bboxes)
        bbox_heights = np.abs(np.diff(bboxes[:,:,1]))
        z_scores = np.abs(zscore(bbox_heights))
        return bboxes[np.where(z_scores < threshold)[0]]        

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
    
    def _get_clustered_character_bboxes(self, num_clusters=None):
        heatmap_rects = self._get_heatmap_character_bboxes()
        
        clusters = self.cluster(heatmap_rects)
        
        
        baselines = np.array([max(y) for y in heatmap_rects[:,:, 1]])
        
        # Need to have more than one character in the cluster for it to be used
        cluster_min = 1
        if num_clusters:
            cluster_labels = hcluster.fclusterdata(baselines.reshape(-1,1), t=3, criterion="maxclust")
        else:
            thresh = 10
            cluster_labels = hcluster.fclusterdata(baselines.reshape(-1,1), thresh, criterion="distance")
        clusters = {}
        for label, rect in zip(cluster_labels, heatmap_rects):
            clusters.setdefault(label, []).append(rect)

        # TODO: Rather than filtering - is it worth reclustering with maxclust?
        return {label: self._filter_outliers(cluster) for label, cluster in clusters.items() if len(cluster) > cluster_min}        
        
    def _visualise(self, image):

        image = self.heatmap.copy()
        clusters = self._get_clustered_character_bboxes()

        for label, cluster in clusters.items():
            color = random_colour()
            for rect in cluster:
                pt1, pt2 = rect
                cv2.rectangle(image, pt1, pt2, color, -1)

        # for text_line in self._get_text_lines():
        #     color = random_colour()
        #     pt1, pt2 = text_line.rect
        #     cv2.rectangle(image, pt1, pt2, color, 1)                
        return image
    
    def visualise_heatmap(self):
        image = self.heatmap.copy()
        return self._visualise(image)
    
    def __len__(self):
        return len(self.text_lines)    