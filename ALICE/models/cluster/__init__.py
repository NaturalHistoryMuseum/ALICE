import numpy as np
import itertools
import pandas as pd
import cv2

from alice.utils import min_max
from alice.utils.geometry import points_to_numpy
from alice.models.geometric import Rectangle, Point

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
        return np.diff(min_max(self._x_coords))[0]

    @property
    def height(self):
        # Get all the x values - width is difference between min and max
        return np.diff(self.interval)[0]
    
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