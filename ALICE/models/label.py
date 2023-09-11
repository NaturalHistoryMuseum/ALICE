import numpy as np
import cv2
import imutils
from typing import List, Literal


from alice.config import logger
from alice.utils import min_max, iter_list_from_value
from alice.utils.geometry import calculate_line_slope_intercept, approx_best_fit_quadrilateral, get_furthest_point_perpendicular_from_line, extend_line, calculate_line_intersecting_point
from alice.models.base import Base
from alice.models.quadrilateral import Quadrilateral
from alice.models.point import Point, points_to_numpy


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
   
    def _get_quadrilateral(self):
        polygon = self.label_mask.get_polygon(epsilon=5)
        vertices = approx_best_fit_quadrilateral(polygon)  
        return Quadrilateral(vertices, self.image)
        
    def get_quadrilateral(self):
        quad = self._get_quadrilateral()
        if quad.is_wellformed_label_shape():
            logger.info("Detected wellformed label quadrilateral")
            return quad
        
        logger.info("Initial quadrilateral is not well formed")
        logger.debug_image(quad.visualise(), 'quad-not-well-formed')
        
        if len(quad.get_invalid_corners()) == 1:
            corrected_quad = self.correct_invalid_corner(quad)
            if corrected_quad.is_wellformed_label_shape():
                logger.info("Using corrected corner quadrilateral")                
                logger.debug_image(corrected_quad.visualise(), 'corner-corrected-quad')
                return corrected_quad
            else:
                logger.info("Correcting corner did not produce well-formed quadrilateral")
                err_vertice = quad.vertices[quad.get_invalid_corners()[0]]
                image = corrected_quad.visualise()
                err_vertice.visualise(image, (255, 0, 255), 20)
                logger.debug_image(image, 'corner-corrected-error-quad')
                                    
        else:
            logger.info("Cannot use corner correction - %s invalid corners", len(quad.get_invalid_corners()))        

        # We use the original quad to guestimate from - not the corrected corner shape
        if quad.nearest_corner_is_good():
            logger.info("Guestimating quad from good nearest corner")            
            approx_quad = self.approx_quadrilateral_from_closest_edges(quad)             
            if approx_quad.is_wellformed_label_shape(): 
                logger.debug_image(approx_quad.visualise(), 'approx-quad-wellformed')
                return approx_quad
            else:
                logger.info("Approximted quad not well formed")
                logger.debug_image(approx_quad.visualise(), 'approx-quad-not-wellformed')
                
        else:
            logger.info("Cannot guesstimate quad - nearest corner not suitable")
            logger.debug_image(quad.visualise(), 'quad-nearest-corner-error')
                            
        self.valid = LabelValid.INVALID_QUADRILATERAL
        
    def get_invalid_corner_edge(self, invalid_corner: Literal['a', 'b', 'c', 'd'], quad):       
        """
        For an invalid corner get the edge to correct
        Look at edges on both side and select the wonkiest - the one with the greatest difference in slopes
        """
        next_edge = [edge for edge in quad.edges.keys() if edge.startswith(invalid_corner)][0]
        edges = list(iter_list_from_value(list(quad.edges.keys()), next_edge))

        parallel_edges = [
            [edges[3], edges[1]],
            [edges[0], edges[2]]
        ]

        slope_diff = []
        for edges in parallel_edges:            
            slope1 = self._get_line_slope(quad.edges[edges[0]])
            slope2 = self._get_line_slope(quad.edges[edges[1]])
            slope_diff.append(abs(slope1 - slope2))

        slope_diff = np.array(slope_diff)    
        invalid_corner_edge = parallel_edges[slope_diff.argmax(axis=0)][0]
        return invalid_corner_edge   
    
    @staticmethod
    def _get_line_slope(line):
        m, _ = calculate_line_slope_intercept(line)        
        return m
    
    def get_invalid_corner(self, quad):
        invalid_corners = quad.get_invalid_corners()
        if len(invalid_corners) == 1:
            return invalid_corners[0]
        else:
            logger.info('Could not retrieve invalid corner - %s invalid corners', len(invalid_corners))

    def correct_invalid_corner(self, quad):
        """
        If we have a single invalid corner, try and correct it using the opposite edge
        As a guide for a new line and intersection points
        """
        corner_labels = list(quad.vertices.keys())
        
        if invalid_corner := self.get_invalid_corner(quad):
            edge = self.get_invalid_corner_edge(invalid_corner, quad)
        
        edges = list(quad.edges.keys())
        idx = edges.index(edge)

        # Get the opposite edge index
        oppos_edge = edges[(idx + 2) % 4]
        adj_edge_1 = edges[(idx - 1) % 4]
        adj_edge_2 = edges[(idx + 1) % 4]

        adj_edge_lines = [quad.edges[adj_edge_1], quad.edges[adj_edge_2]]
        oppos_edge_line = quad.edges[oppos_edge]

        # Get futhest mask point perpendicular to the oppos edge
        furthest_point = get_furthest_point_perpendicular_from_line(oppos_edge_line, self.label_mask.edge_points())
        # ...and create a new line insecting this point
        new_edge = calculate_line_intersecting_point(oppos_edge_line, furthest_point)
        
        correct_corners = set(corner_labels) - set(edge.split('_'))
        new_vertices = [Point(*quad.vertices[v]) for v in correct_corners]
        
        for adj_edge in adj_edge_lines:
            intersection = extend_line(adj_edge).intersection(new_edge)    
            new_vertices.append(Point.from_shapely(intersection))     
            
        return Quadrilateral(new_vertices, self.image)
            
    def approx_quadrilateral_from_closest_edges(self, quad):
        """
        Approximate the quadrilateral vertices, from two edges
        
        We create a line with same slope, and then calculate the position intersecting
        the mask at the greatest perpendicular distance from the initial edge
        
        Intersection of these four lines will be the quadrilateral vertices
        
        """
        
        # Closest point is a - width edges a_b and d_a
        closest_edges = [quad.edges[e] for e in ['a_b', 'd_a']]
        
        lines = []
        # Loop through the two good edges, creating a line with the same slope
        # which interesect the further perpendicular edge of the mask
        for edge in closest_edges:
            extended_edge = extend_line(edge, self.image_width * self.image_height)
            furthest_point = get_furthest_point_perpendicular_from_line(extended_edge, self.label_mask.edge_points())
            new_line = calculate_line_intersecting_point(extended_edge, furthest_point)                        
            lines.append((extended_edge, new_line))
    
        # Calculate new vertices from line intersections
        new_vertices = []
        for line in lines[0]:
            for perpendicular_line in lines[1]:
                if intersection := line.intersection(perpendicular_line):
                    new_vertices.append(Point.from_shapely(intersection))  

        # FIXME: The accuracy of this can be improved by adding perspective correction    
        
        return Quadrilateral(new_vertices, self.image)

    def _visualise(self, image):
        if self.quad:
            return self.quad.visualise(image)

        # We don't have a quad, so label won't be used - mark with red top
        cv2.rectangle(image, (0, 0), (self.image_width, 10), (255, 0, 0), -1)

        for point in self.vertices:
            point.visualise(image)

        return image

    def set_valid(self, valid: LabelValid):
        logger.info("Setting label as validity: %s", valid)
        logger.debug_image(self.visualise(), f'label-invalid-{valid.name.lower()}')   
        self.valid = valid

    def is_valid(self):
        return self.valid == LabelValid.VALID

    def crop(self, max_shortest_edge, max_longest_edge):
        
        quad = self.get_quadrilateral()     
        
        if not self.is_valid():
            logger.info("Label is invalid - no crop available")
            return
        elif not quad:
            logger.info("No quad for label - no crop available")
            return

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

        src = points_to_numpy(self.quad.vertices.values())
        M = cv2.getPerspectiveTransform(src.astype(np.float32), dest)
    
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

    def count_valid(self):
        return len(list(self.iter_valid()))
    
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
                logger.info('Min Max: %s', min_maxes)             
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
            # Create an array of True True so we keep both
            mask = np.array([True] * 2)
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
        max_deviation = np.mean(data) / len(data)
        # FIXME: Why is one of the labels here being disallowed??
        print(max_deviation)
        print(deviation)
        print(deviation < max_deviation)
        # re sub for corner label
        return deviation < max_deviation    