import numpy as np
import cv2
from collections import OrderedDict
from shapely import LineString
import cv2
from typing import List
from enum import Enum

from alice.utils import iter_list_from_value, pairwise, min_max
from alice.utils.geometry import calculate_angle, order_points
from alice.models.base import Base
from alice.models.point import Point, points_to_numpy
from alice.config import logger

class QuadMethod(Enum):
    DEFAULT = 0
    CORNER_CORRECTED = 1
    CORNER_PROJECTED = 2

class Quadrilateral(Base):
    
    # Corners must be in this range        
    valid_angle = {
        'min': 30,
        'max': 160
    }    
    
    def __init__(self, vertices: List[Point], image, method=QuadMethod.DEFAULT):
        super().__init__(image)
        self.method = method
        
        vertices = order_points(vertices)
        
        # Closest point to the bottom of the canvas
        self.closest_point = self.get_closest_point(
            points_to_numpy(vertices)
        )

        # Vertices are assigned to a,b,c,d
        # A will be the closest point (with the maximum y value), with the points
        # ordered counter clockwise, with b being the next corner counter clockwise
        # A & C will be opposite each other; B & D opposite
        self.vertices = OrderedDict(zip(['a', 'b', 'c', 'd'], iter_list_from_value(vertices, self.closest_point)))

        # Loop through vertices, creating edges names a_b, b_c etc.,
        self.edges = OrderedDict([(
            f'{k1}_{k2}', LineString([
                self.vertices[k1], 
                self.vertices[k2]
            ])) for k1, k2 in pairwise(list(self.vertices.keys()))
        ])
        self.angles = self.get_corner_angles()

    def get_closest_point(self, vertices):
        """
        Closest point is the bottom corner nearest the center point of the image
        """
        center = round(self.image_width / 2)
        vertices = np.array(vertices)
        bottom_corners = vertices[np.argpartition(vertices[:, 1], -2)[-2:]]
        x_offset_from_center = np.abs(bottom_corners[:, 0] - center)
        closest_point = bottom_corners[np.argmin(x_offset_from_center)]
        return Point(*closest_point)

    def get_corner_angles(self):
        angles = {}
        maxv = len(self.vertices) - 1
        vertice_labels = list(self.vertices.keys())
        for i, vertice in enumerate(vertice_labels):    
            next_vertice = vertice_labels[i+1 if i < maxv else 0]
            prev_vertice = vertice_labels[i-1 if i > 0 else maxv]
            vertices = points_to_numpy([
                self.vertices[prev_vertice],
                self.vertices[vertice],
                self.vertices[next_vertice]
            ])
            angle = calculate_angle(*vertices.ravel())
            angles[vertice] = angle        
        return angles

    def is_wellformed_label_shape(self):
        """
        Validate the label shape, 4 corners & opposite angles be within 15 degrees
        """

        # Oppos corners should not have more than this angle difference
        # Otherwise box is squashed
        valid_angle_diff = 20
        if len(self.vertices) != 4:
            return False
            
        angles = np.array(list(self.angles.values()))
        min_angle, max_angle = min_max(angles)
        angle_diff = self.get_max_angle_diff()
        
        if min_angle < self.valid_angle['min']:
            logger.info('Quadrilateral min angle %s is below threshold of %s', min_angle, self.valid_angle['min'])
            return False
        elif max_angle > self.valid_angle['max']:
            logger.info('Quadrilateral max angle %s is above threshold of %s', max_angle, self.valid_angle['max'])
            return False
        elif(angle_diff > valid_angle_diff):
            logger.info('Quadrilateral angle diff %s is above threshold of %s', angle_diff, valid_angle_diff)
            return False
        
        return True
    
    def get_max_angle_diff(self):
        oppos_corners = [('a', 'c'), ('b', 'd')]
        return max([abs(self.angles[i] - self.angles[j]) for i,j in oppos_corners])         

    def nearest_corner_is_good(self):
        """
        See if the nearest corner has acceptable angle range
        """
        # Stricter upper bounds than the valid_angle above; all other corners are based
        # on thise one, so it needs to be good
        lower_bound = 100
        upper_bound = 130
        return lower_bound <= self.angles['a'] <= upper_bound
    
    def _is_valid_angle(self, angle: int):
       return self.valid_angle['min'] <= angle <= self.valid_angle['max']
            
    def get_invalid_corners(self):            
        return [label for label, angle in self.angles.items() if not self._is_valid_angle(angle)]

    @property
    def x_length(self):
        return round(max([self.edges['a_b'].length, self.edges['c_d'].length]))

    @property
    def y_length(self):
        return round(max([self.edges['b_c'].length, self.edges['d_a'].length]))

    def _visualise(self, image):
        if self.method == QuadMethod.DEFAULT:
            edge_color = (0, 255, 0)
        elif self.method == QuadMethod.CORNER_CORRECTED:
            edge_color = (255, 0, 0)
        elif self.method == QuadMethod.CORNER_PROJECTED:
            edge_color = (0, 0, 255)            
        
        for edge in self.edges.values():
            p = np.array(edge.coords).astype(np.int32)
            cv2.line(image, p[0], p[1], edge_color, 5)
            
        for point in self.vertices.values():
            point.visualise(image)

        self.closest_point.visualise(image, (255,0,255))
        
        return image