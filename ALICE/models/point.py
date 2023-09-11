import numpy as np
from shapely import Point as ShapelyPoint
from typing import List
import cv2

class Point:
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)
        
    @classmethod
    def from_shapely(cls, point: ShapelyPoint):
        """
        Create a new instance of this class from a shapely point
        """
        x,y = list(point.coords)[0]
        return cls(x, y)

    def __iter__(self):
        """
        Allow unpacking to x, y = point
        """
        yield self.x
        yield self.y

    def to_numpy(self):
        return np.array([self.x, self.y])

    def to_tuple(self):
        return (self.x, self.y)
    
    def visualise(self, image, colour=(255,0,0), size=10):
        cv2.circle(image, self.to_tuple(), 5, colour, size)
        return image
    
    def __getitem__(self, i):
        xy = self.to_numpy() 
        return xy[i]

    def __repr__(self):
        return f'Point({self.x}, {self.y})'

    def __eq__(self, other): 
        return self.x == other.x and self.y == other.y    
    
    
def points_to_numpy(points: List[Point]) -> np.array:
    """
    Convert list of Points to numpy array
    """
    return np.array([point.to_numpy() for point in points])    