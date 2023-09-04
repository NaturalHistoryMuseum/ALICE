import numpy as np
import cv2
import imutils


from alice.config import logger
from alice.utils import min_max
from alice.utils.geometry import approx_best_fit_ngon, get_furthest_point_perpendicular_from_line, extend_line, get_line_at_point
from alice.models.base import Base
from alice.models.quadrilateral import Quadrilateral

from enum import Enum

class LabelValid(Enum):
    VALID = 0
    INVALID_QUADRILATERAL = 1
    NONDETECTED_LABELS = 2
    INVALID_SHAPE = 3
    
class InvalidLabel():
    
    """
    Helper class for creating an invalid label so order of labels can be preserved
    """

    valid = LabelValid.NONDETECTED_LABELS

    def is_valid(self):
        return False
        

class Label(Base):

    """
    Representing a single label
    """

    def __init__(self, label_mask, image):
        self.valid = LabelValid.VALID
        super().__init__(image)
        self.label_mask = label_mask
        polygon = label_mask.get_polygon(epsilon=5)
        self.vertices = approx_best_fit_ngon(polygon)        
        self.quad = self._get_quadrilateral()        

    def _get_quadrilateral(self):

        quad = Quadrilateral(self.vertices, self.image)
        if quad.is_wellformed_label_shape():
            return quad

        logger.info("Quad not well formed shape")
        logger.debug_image(quad.visualise(), 'not-wellformed-quad')
        
        if quad.nearest_corner_is_good():
            logger.info("Guestimating quad from good nearest corner")
            closest_edges = [quad.edges[e] for e in ['a_b', 'd_a']]
            vertices = self.approx_quadrilateral_from_closest_edges(closest_edges) 
            approx_quad = Quadrilateral(vertices, self.image, is_approx=True)
            
            if approx_quad.is_wellformed_label_shape(): 
                logger.debug_image(approx_quad.visualise(), 'approx-quad-wellformed')
                return approx_quad
            else:
                logger.info("Approximted quad not well formed")
                
        else:
            logger.info("Cannot guesstimate quad - nearest corner not suitable")
        
        self.set_valid(LabelValid.INVALID_QUADRILATERAL)

    def approx_quadrilateral_from_closest_edges(self, edges):
        """
        Approximate the quadrilateral vertices, from two edges
        
        We create a line with same slope, and then calculate the position intersecting
        the mask at the greatest perpendicular distance from the initial edge
        
        Intersection of these four lines will be the quadrilateral vertices
        
        """
        lines = []
        # Loop through the two good edges, creating a line with the same slope
        # which interesect the further perpendicular edge of the mask
        for edge in edges:
            extended_edge = extend_line(edge, self.image_width * self.image_height)
            furthest_point = get_furthest_point_perpendicular_from_line(extended_edge, self.label_mask.edge_points())
            new_line = get_line_at_point(extended_edge, furthest_point, self.image_width)
            lines.append((extended_edge, new_line))
    
        # Calculate new vertices from line intersections
        vertices = []
        for line in lines[0]:
            for perpendicular_line in lines[1]:
                if intersection := line.intersection(perpendicular_line):
                    vertices.append((int(intersection.x), int(intersection.y)))  
    
        # FIXME: The accuracy of this can be improved by adding perspective correction    
        ordered_vertices = cv2.convexHull(np.array(vertices), clockwise=False, returnPoints=True)
        # Convert to list of tuples
        return [tuple(point[0]) for point in ordered_vertices]            

    def _visualise(self, image):
        if self.quad:
            return self.quad.visualise(image)

        # We don't have a quad, so label won;t be used - mark with red top
        cv2.rectangle(image, (0, 0), (self.image_width, 10), (255, 0, 0), -1)

        for point in self.vertices:
            pt = np.array(point).astype(np.int32)
            cv2.circle(image, pt, 5, (255,0,0), 10)

        return image

    def set_valid(self, valid: LabelValid):
        logger.info("Setting label as validity: %s", valid)
        self.valid = valid

    def is_valid(self):
        return self.valid == LabelValid.VALID

    def crop(self, max_shortest_edge, max_longest_edge):

        x_is_longest = self.quad.x_length > self.quad.y_length
        
        if x_is_longest:
            x = max_longest_edge
            y = max_shortest_edge
        else:
            x = max_shortest_edge
            y = max_longest_edge
            
        dest = np.float32([
            (0, x), #A
            (0, 0), #B
            (y, 0), #C
            (y, x) #D
        ])
        
        src = np.float32(list(self.quad.vertices.values()))
        M = cv2.getPerspectiveTransform(src, dest)
    
        cropped = cv2.warpPerspective(self.image, M,(y, x),flags=cv2.INTER_LINEAR)   
        return cropped
    
    
class LabelQuartet:
    """
    The four labels
    """
    def __init__(self):
        self._labels = []
        
    def add_label(self, label):
        self._labels.append(label)

    def get_cropped_labels(self):
        max_shortest_edge, max_longest_edge = self.get_dimensions()
        first_landscape = 0
        cropped = {}
        for i, label in enumerate(self._labels):
            if not label.is_valid(): continue

            cropped_image = label.crop(max_shortest_edge, max_longest_edge)
            h, w = cropped_image.shape[:2]
            is_landscape = w > h
            # Capture the first landscape label - we'll use this as 
            # the base to rotate to            
            if not first_landscape and is_landscape:
                first_landscape = i

            cropped[i] = label.crop(max_shortest_edge, max_longest_edge)

        # Rotate all images to the first landscape one
        rotated = {}
        for i in range(len(self._labels)):
            if not i in cropped: continue
            rotation = (i - first_landscape) * 90
            rotated[i] = imutils.rotate_bound(cropped[i], rotation)

        return rotated

    def iter_valid(self):
        for label in self._labels:
            if label.is_valid():
                yield label

    def _get_dimensions_min_max(self):
        dimensions = np.array([
            (label.quad.x_length, label.quad.y_length) for label in self.iter_valid()
        ])
        return np.array([min_max(d) for d in dimensions]) 
        
    def get_dimensions(self):
        """
        Get edge dimensions for all labels in the view
        """
        min_maxes = self._get_dimensions_min_max()
        min_maxes = self.validate_shape_homogeneity(min_maxes)
        return np.max(min_maxes[:,0]), np.max(min_maxes[:,1])        

    def validate_shape_homogeneity(self, min_maxes):

        if len(min_maxes) <= 1:
            return min_maxes
            
        if len(min_maxes) == 2:
            outliers_mask = self._validate_shape_difference(min_maxes)
        else:
            outliers_mask = self._validate_shape_deviation(min_maxes)
            
        # Mark these labels as invalid so they won't be included in the merge
        for i, label in enumerate(self.iter_valid()):
            if not outliers_mask[i]:
                logger.info(f"Label {i} is not within normal range of other labels ")
                label.set_valid(LabelValid.INVALID_SHAPE)
            
        # Mask the min maxes value, so outlines won't be used in dimension calculations 
        return min_maxes[outliers_mask]              

    def _validate_shape_difference(self, min_maxes):
        """
        Validate shape lengths are not two disimilar from each other
        Used for validating shapes when we just have two labels 
        """
        min_diff = np.mean(min_maxes) / 4
        # Calculate the maxium different along the edges
        diff = np.max(np.diff(min_maxes, axis=1))
        if diff > min_diff:
            # Create a mask where True is set to the highest value
            max_value = np.max(min_maxes[:, 1])
            mask = min_maxes[:, 1] == max_value
        else:
            mask = np.any(min_diff, axis=1)
        return mask
            
    def _validate_shape_deviation(self, min_maxes):
        """
        Validate the deviations in shape, and remove outliers 
        """
        # Loop through the min max columns, checking they are within accepted deviations
        outliers = np.array([self._get_outliers(data) for data in min_maxes.T])
        # Combine both array of outliers using logical and into an outliers_mask
        return np.logical_and(outliers[0], outliers[1])    
    
    def _get_outliers(self, data):
        # Calculate median deviation
        median = np.median(data)
        # calculate median absolute deviation
        deviation = np.sqrt((data - median)**2)
        max_deviation = np.mean(data) / 4
        return deviation < max_deviation    