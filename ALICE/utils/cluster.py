import numpy as np
from scipy.stats import zscore
import itertools
import pandas as pd
import cv2

from alice.utils import min_max
from alice.config import logger
from alice.utils.geometry import points_to_numpy
from alice.models.geometric import Rectangle, Point

class Cluster():
    
    # If the cluster width isn't more than 50% of the image, it's invalid
    min_width_ratio = 0.3
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
        return np.diff(min_max(self._x_coords))[0]

    @property
    def height(self):
        # Get all the x values - width is difference between min and max
        return np.diff(self.interval)[0]

    @property
    def top(self):
        return np.min(self._y_coords)

    @property
    def bottom(self):
        return np.max(self._y_coords)
    
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
            logger.info('Invalid cluster - too few characters (%s)', len(self.bboxes))
            return False
        width_ratio = self.width / image_width
        if width_ratio < self.min_width_ratio:
            logger.info('Invalid cluster - width %s below min width ration %s < %s', self.width, width_ratio, self.min_width_ratio)
            return False        
        if self.has_xaxis_overlaps():
            logger.info('Invalid cluster - has xaxis overlaps')
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
    Cluster boxes horizontal axis, using y coords as intervals
    If a bbox intersects multiple, then use the cluster with greatest intersection
    """

    def __call__(self, bboxes: np.array):
        bboxes = self._sort_horizontally(bboxes)
        bboxes = self._filter_outliers(bboxes)
        return self._cluster(bboxes)        

    def _sort_horizontally(self, bboxes: np.array):
        return bboxes[bboxes[:, 0, 0].argsort()]

    def _get_best_intersection(self, intersections):
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
                intersections = np.array(intersections)
                # If we have multiple intersections, check they aren't caused by a conjoined 
                # box (with height similar to height of intersected clusters) 
                if len(intersections) > 1:
                    bbox_height = np.diff(min_max(bbox[:,1]))[0]
                    # Get the height of the last two boxes of the cluster
                    # NOTE: If we use the cluster height, this can fail to detect conjoined chars as the 
                    # cluster is to high to detect them
                    last_boxes_height = np.diff(min_max(np.array([clusters[i].last_bbox_interval for i in intersections[:, 0]])))[0]      
                    r = bbox_height / last_boxes_height
                    # Conjoined character box spanning multiple clusters                    
                    if r > 0.8:
                        logger.info('Bounding box height %s is similar to height %s of %s of last boxes in intersected clusters. Skipping - possibly conjoined characters ', bbox_height, last_boxes_height, len(intersections))
                        continue
                 
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