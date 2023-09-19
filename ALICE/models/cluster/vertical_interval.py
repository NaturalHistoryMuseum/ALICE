import numpy as np
from scipy.stats import zscore

from alice.utils import min_max
from alice.config import logger
from alice.models.cluster import Cluster


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