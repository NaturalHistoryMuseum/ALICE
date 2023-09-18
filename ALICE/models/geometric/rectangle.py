import cv2
from shapely import Polygon
import numpy as np

from alice.models.geometric.point import Point
from alice.utils.geometry import points_to_numpy

class Rectangle:
    
    def __init__(self, top_left, bottom_right):
        self.top_left = top_left
        self.bottom_right = bottom_right
        self.polygon = self._to_polygon()

    @property
    def area(self):
        return self.polygon.area

    @property
    def yy(self):
        return [self.top_left[1], self.bottom_right[1]]

    @property
    def xx(self):
        return [self.top_left[0], self.bottom_right[0]]
    
    @property
    def width(self):
        return self.bottom_right[0] - self.top_left[0]
    
    @property
    def height(self):
        return self.bottom_right[1] - self.top_left[1]

    @property
    def vertices(self):
        top_right = Point(self.bottom_right[0], self.top_left[1])
        bottom_left = Point(self.top_left[0], self.bottom_right[1])
        return [self.top_left, top_right, self.bottom_right, bottom_left] 
        
    @classmethod
    def from_numpy(cls, arr: np.array):
        """
        Create a new instance from numpy array
        """
        tl = Point(*arr[0])
        br = Point(*arr[1])
        return cls(tl, br)
    
    def _to_polygon(self):
        return Polygon(points_to_numpy(self.vertices))

    def as_numpy(self):        
        return points_to_numpy([self.top_left, self.bottom_right])
        
    def intersection(self, rectangle):
        return self.polygon.intersection(rectangle.polygon)

    def visualise(self, image, colour=(255,0,0), size=2):
        cv2.rectangle(image, self.top_left.to_tuple(), self.bottom_right.to_tuple(), colour, size)
        return image        

    def __repr__(self):
        return f'Rectangle({self.top_left.to_tuple()}, {self.bottom_right.to_tuple()})'