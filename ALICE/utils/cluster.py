
from scipy.stats import zscore
import numpy as np
import os
import cv2
import re
import imutils
import pandas as pd

from alice.utils import min_max


class Cluster():

    def __init__(self, bbox):
        self._bboxes = []
        self.add(bbox)

    @property
    def interval(self):
        return min_max(self._y_intervals)

    @property
    def last_bbox_interval(self):
        return min_max(self._y_intervals[-1])
 
    @property
    def _y_intervals(self):
        # Get all the y value of the boxes
        return np.array(self._bboxes)[:,:,1]
    
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

    def __len__(self):
        return len(self._bboxes)

    def __getitem__(self, i):
        return self._bboxes[i]      
    
    def __iter__(self):
        yield from self._bboxes      

    def __repr__(self):
        return f'Cluster({self.top}, {self.bottom})'


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
        
        for bbox in bboxes:
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
                
        return clusters        

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