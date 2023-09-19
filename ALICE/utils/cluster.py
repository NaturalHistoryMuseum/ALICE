
from scipy.stats import zscore
import numpy as np
import os
import cv2
import re
import imutils
import pandas as pd
import itertools

from alice.utils import min_max
from alice.models.geometric import Rectangle, Point
from alice.utils.geometry import points_to_numpy

class ClusterBoundingBox(Base):
    """
    Get a cluster around a bounding box

    all_line_rects - List of all lines of text, used for padding bounding box
    
    """
    
    def __init__(self, cluster:Cluster, image:np.array, all_line_rects:List[Rectangle]):
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
            

class Cluster():
    
    # If the cluster width isn't more than 50% of the image, it's invalid
    min_width_ratio = 0.5
    # If cluster has less than 6 bboxes, it's invalid
    min_bbox_num = 4

    def __init__(self, bbox):
        self._bboxes = []
        self.add(bbox)

    @property
    def interval(self):
        return min_max(self._y_coords)

    @property
    def last_bbox_interval(self):
        return min_max(self._y_coords[-1])
 
    @property
    def width(self):
        # Get all the x values - width is difference between min and max
        return np.diff(min_max(self.bboxes[:,:,0]))[0]
 
    @property
    def _y_coords(self):
        # Get all the y value of the boxes
        return self.bboxes[:,:,1]
        
    @property
    def _x_coords(self):
        # Get all the x values of the boxes
        return self.bboxes[:,:,0]        
        
    @property
    def bboxes(self):
        # Get bboxes as numpy array
        return np.array(self._bboxes)
        
    @property
    def hull(self):    
        points = list(itertools.chain(*[Rectangle.from_numpy(bbox).vertices for bbox in self.bboxes]))
        return cv2.convexHull(points_to_numpy(points))       
    
    def add(self, bbox):
        self._bboxes.append(bbox)

    def intersects(self, bbox):
        top, bottom = min_max(bbox[:,1])  
        return pd.Interval(top, bottom).overlaps(pd.Interval(*self.interval))

    def intersection(self, bbox):   
        """
        Calculate both the cluster intersection, and the intersection with the 
        Last bounding box - many lines of text curve so intersection with last boundng box
        Is a better feature to cluster on
        """
        box_int = self._intersection(bbox, *self.last_bbox_interval)
        cluster_int = self._intersection(bbox, *self.interval)
        return box_int, cluster_int

    def _intersection(self, bbox, top, bottom):
        bbox_top, bbox_bottom = min_max(bbox[:,1])
        return max(min([bbox_bottom, bottom]) - max([bbox_top, top]), 0) 
    
    def get_bounding_box(self):
        """
        Get a bounding box around the elements in this cluster
        """
        ymin, ymax = min_max(self.bboxes[:,:,1])
        xmin, xmax = min_max(self.bboxes[:,:,0])
        return Rectangle(Point(xmin, ymin), Point(xmax, ymax))
    
    def is_valid(self, image_width):

        if len(self.bboxes) < self.min_bbox_num:
            return False
        if (self.width / image_width) < self.min_width_ratio:
            return False        
        if self.has_xaxis_overlaps():
            return False
        return True
    
    def has_xaxis_overlaps(self):
        """
        Are there any overlaps on the xaxis
        """
        # Get the pairs of x coords, sorted from left to right
        x_coords = self._x_coords[np.argsort(self._x_coords[:, 0])]
        intervals = [pd.Interval(*x) for x in x_coords]
        # Loop through intervals comparing pair by pair for overlaps
        # This works are they are sorted left to right
        return any([i1.overlaps(i2) for i1, i2 in itertools.pairwise(intervals)])

    def __len__(self):
        return len(self._bboxes)

    def __getitem__(self, i):
        return self._bboxes[i]      
    
    def __iter__(self):
        yield from self._bboxes      

    def __repr__(self):
        return f'Cluster({self.interval})'


class ClusterVerticalInterval():
    """
    Cluster boxes horizontally, using y coords as intervals
    If a bbox intersects multiple, then use the cluster with greatest intersection
    """

    def __call__(self, bboxes: np.array):
        bboxes = self._sort_horizontally(bboxes)
        bboxes = self._filter_outliers(bboxes)
        return self._cluster(bboxes)        

    def _sort_horizontally(self, bboxes: np.array):
        return bboxes[bboxes[:, 0, 0].argsort()]

    def _get_best_intersection(self, intersections):
        intersections = np.array(intersections)
        # Sort intersections, so greates box intersections are at the top, and then if it's a 'draw'
        # cluster intersection will be used
        sorted_intersections = intersections[np.lexsort([intersections[:, 2], intersections[:, 1]])[::-1]]
        return sorted_intersections[0][0]
    
    def _cluster(self, bboxes: np.array):        
        # Create an initial cluster for the first box
        clusters = [Cluster(bboxes[0])]
        
        # Skip first bbox - it's already been added to make the first cluster
        for bbox in bboxes[1:]:
            intersections = []
            for cluster_idx, cluster in enumerate(clusters):
                if cluster.intersects(bbox):
                    box_int, cluster_int = cluster.intersection(bbox)
                    intersections.append([cluster_idx, box_int, cluster_int])

            if intersections:                                
                cluster_idx = self._get_best_intersection(intersections)
                clusters[cluster_idx].add(bbox)
            else:
                clusters.append(Cluster(bbox))
                
        return sorted(clusters, key=lambda x: x.interval[0])      

    @staticmethod
    def _interval_intersection(inter1, inter2):
        # Not left right - top bottom
        n = np.array([
            (inter1.left, inter1.right),
            (inter2.left, inter2.right)
        ])      
        intersection = np.min(n[:,1]) - np.max(n[:,0])
        # Note: intersection can be zero if y coords for upper lower edge are the same
        # eg. bottom 205, top 205 => intersection 0
        return intersection, np.min(n), np.max(n)

    @staticmethod
    def _filter_outliers(bboxes):
        """
        Filter any outlier bboxes by height (letters conjoined to top row etc.,) and area (periods etc.,)
        """
        threshold = 3
        bboxes = np.array(bboxes)
        bbox_heights = np.abs(np.diff(bboxes[:,:,1]))
        z_scores = np.abs(zscore(bbox_heights))
        bboxes = bboxes[np.where(z_scores < threshold)[0]]  

        # Filter any boxes with tiny areas - this can through off the intersection calculations
        bbox_areas = np.array([np.prod(np.diff(bbox, axis=0)) for bbox in bboxes])
        # Lower bound is 0.2 of the median - any box with a smaller area will be filtered out
        lower_bound = np.mean(bbox_areas) * 0.2
        bboxes = bboxes[(bbox_areas > lower_bound)]
        
        return bboxes