import numpy as np
from typing import List
import cv2

from alice.utils import random_colour, min_max
from alice.models.base import Base
from alice.models.geometric import Rectangle, Point
from alice.models.text.line import TextLine
from alice.models.cluster import Cluster



class ClusterBoundingBox(Base):
    """
    Get a cluster around a bounding box

    all_line_rects - List of all lines of text, used for padding bounding box
    
    """
    
    def __init__(self, cluster: Cluster, image:np.array, all_line_rects:List[Rectangle]):
        super().__init__(image)
        self._cluster = cluster
        self.rect = self._get_bounding_rect(all_line_rects)

    @property
    def is_valid(self):
        return self._cluster.is_valid(self.image_width)
    
    def _get_centroids(self):
        # Calculate the baseline centroids of the heatmap (letter) rectangles
        return np.array([
            (np.mean(bbox[:, 0], dtype=np.int32), max(bbox[:, 1])) for bbox in self._cluster.bboxes
        ])        
        
    def textline(self):
        return TextLine(self.rect, self.image, self._get_centroids())
        
    def _get_bounding_rect(self, all_line_rects):
        """
        Get a bounding box around a cluster
        """
        cluster_bbox = self._cluster.get_bounding_box()
        padded_ymin, padded_ymax = self._bbox_get_padding(cluster_bbox, all_line_rects)
        return Rectangle(Point(0, padded_ymin), Point(self.image_width, padded_ymax))        
    
    def _bbox_get_padding(self, cluster_bbox, all_line_rects):  
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
                
        for line_rect in all_line_rects:                
            intersection = cluster_bbox.intersection(line_rect)
            r = max([intersection.area / line_rect.area, intersection.area / cluster_bbox.area])
            if r > 0.75:
                # Only extend if new cluster height is line height is within .3 of the original
                if line_rect.height < (cluster_bbox.height * 1.3):
                    yy.extend(line_rect.yy)
        
        return min_max(np.array(yy))

    def _visualise(self, image):
        colour = random_colour()
        for rect in self._cluster:
            pt1, pt2 = rect
            cv2.rectangle(image, pt1, pt2, colour, 1)  
      
        for point in self._get_centroids():
            cv2.circle(image, point, 2, (255, 255, 0), -1)  