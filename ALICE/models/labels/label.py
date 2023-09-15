import numpy as np
import cv2
import imutils
import math
from typing import List, Literal


from alice.config import logger
from alice.utils import iter_list_from_value
from alice.utils.geometry import calculate_line_slope_intercept, points_to_numpy, approx_best_fit_quadrilateral, get_furthest_point_perpendicular_from_line, extend_line, calculate_line_intersecting_point
from alice.models.base import Base
from alice.models.geometric import Quadrilateral, QuadMethod, Point

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
        self._visualisation = image.copy()        
        self.quad = self._get_quadrilateral()        
        if self.quad:
            logger.info("Label quadrilateral detected using %s", self.quad.method.name) 
        else:
            logger.info('No quadrilateral found for label')
                       
    def _get_best_fit_vertices(self):
        polygon = self.label_mask.get_polygon(epsilon=5)
        return approx_best_fit_quadrilateral(polygon)  
    
    def _get_best_fit_polygon_quadrilateral(self):
        vertices = self._get_best_fit_vertices()    
        return Quadrilateral(vertices, self.image)    
        
    def _get_quadrilateral(self):
        
        quad = self._get_best_fit_polygon_quadrilateral()
        self._visualisation = quad.visualise(self._visualisation)
            
        if quad.is_wellformed_label_shape():                                    
            return quad
        
        logger.info("Best fit polygon quad not well formed") 
        if len(quad.get_invalid_corners()) == 1:
            corrected_quad = self._get_corner_corrected_quadrilateral(quad)
            # self._visualisation = corrected_quad.visualise(self._visualisation)
            if corrected_quad.is_wellformed_label_shape():
                return corrected_quad
        else:
            logger.info("Cannot use corner correction - %s invalid corners", len(quad.get_invalid_corners()))        

        # We use the original quad to guestimate from - not the corrected corner shape
        if quad.nearest_corner_is_good():
            logger.info("Guestimating quad from good nearest corner")            
            projected_quad = self._project_quadrilateral_from_closest_edges(quad)    
            self._visualisation = projected_quad.visualise(self._visualisation)         
            if projected_quad.is_wellformed_label_shape(): 
                return projected_quad
                            
        self.valid = LabelValid.INVALID_QUADRILATERAL
        
    def get_angle_of_parallel_edges(self, quad, next_edge):
        edges = list(iter_list_from_value(list(quad.edges.keys()), next_edge))
        parallel_edges = [
            [edges[3], edges[1]],
            [edges[0], edges[2]]
        ]
        angles = {}   
        for edges in parallel_edges:                
            edge1 = quad.edges[edges[0]]
            edge2 = quad.edges[edges[1]]   
            angles[edges[0]] = self._get_angle_between_lines(edge1, edge2)
        return angles

    def get_invalid_corner_edge(self, invalid_corner: Literal['a', 'b', 'c', 'd'], quad):       
        """
        For an invalid corner, identify the edge to correct
        
        First, we try and find the 'wonkiest' - the side with the greatest difference in angle to its 
        opposite/parallel edge
        
        If both parallel edges have a similar angle (must be at least 50% different), we triangulate to the 
        centroid of the mask - the line with the points with greatest difference from the centroid will be corrected

        """
        
        vertice_labels = list(quad.vertices.keys())
        idx = vertice_labels.index(invalid_corner)
        prev_corner = vertice_labels[(idx - 1) % 4]
        next_corner = vertice_labels[(idx + 1) % 4]
        # Next edge after the invalid corner
        next_edge = f'{invalid_corner}_{next_corner}'
        prev_edge = f'{prev_corner}_{invalid_corner}'
                
        parallel_edge_angles = self.get_angle_of_parallel_edges(quad, next_edge)
        
        angles_arr = np.array(list(parallel_edge_angles.values()))
        # The parallel angles need to be sufficiently different to be used
        if np.abs(np.diff(angles_arr)) / np.max(angles_arr) > 0.5:
            return max(parallel_edge_angles, key=lambda k: parallel_edge_angles[k])

        edge_labels = [
            prev_edge,
            next_edge
        ]
        centroid = self.label_mask.centroid
        dist_diff = [] 
                            
        for edge_label in edge_labels:
            edge = quad.edges[edge_label]
            dist_from_centroid = np.array([math.dist(pt, centroid) for pt in edge.coords])
            dist_diff.append(np.abs(np.diff(dist_from_centroid))[0])

        dist_diff = np.array(dist_diff)
        invalid_corner_edge = edge_labels[dist_diff.argmax(axis=0)]
        return invalid_corner_edge

    @staticmethod
    def _get_line_slope(line):
        m, _ = calculate_line_slope_intercept(line)        
        return m
    
    @staticmethod
    def _get_angle_between_lines(line1, line2):
        m1, _ = calculate_line_slope_intercept(line1)    
        m2, _ = calculate_line_slope_intercept(line2)      
        return abs(math.degrees(math.atan((m2-m1)/(1+(m2*m1)))))
    
    def get_invalid_corner(self, quad):
        invalid_corners = quad.get_invalid_corners()
        if len(invalid_corners) == 1:
            return invalid_corners[0]
        else:
            logger.info('Could not retrieve invalid corner - %s invalid corners', len(invalid_corners))

    def _get_corner_corrected_quadrilateral(self, quad):
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
            
        return Quadrilateral(new_vertices, self.image, QuadMethod.CORNER_CORRECTED)
            
    def _project_quadrilateral_from_closest_edges(self, quad):
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
                    
        return Quadrilateral(new_vertices, self.image, QuadMethod.CORNER_PROJECTED)


    def set_valid(self, valid: LabelValid):
        logger.info("Setting label validity: %s", valid)
        self.valid = valid

    def is_valid(self):
        return self.valid == LabelValid.VALID

    def crop(self, max_shortest_edge, max_longest_edge):

        if not self.is_valid():
            logger.info("Label is invalid - no crop available")
            return
        
        elif not self.quad:
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
     

    def _visualise(self, image):
        
        if not self.quad:
            # We don't have a quad, so label won't be used - mark with red top
            cv2.rectangle(self._visualisation, (0, 0), (self.image_width, 10), (255, 0, 0), -1)

            for point in self._get_best_fit_vertices():
                point.visualise(self._visualisation)

        return self._visualisation     