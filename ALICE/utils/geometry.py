import math
import numpy as np
import cv2
from copy import deepcopy
from scipy.interpolate import interp1d
from skimage import measure
from collections import Counter
from shapely import LineString
import sympy
import mpmath
from typing import List

from alice.models.geometric import Point
from alice.config import IMAGE_BASE_WIDTH

def calculate_angle(x1, y1, x2, y2, x3, y3):
    """
    Calculate angle between three points
    """
    vector1 = [x1 - x2, y1 - y2]
    vector2 = [x3 - x2, y3 - y2]
    
    dot_product = sum(i * j for i, j in zip(vector1, vector2))
    magnitude1 = math.sqrt(sum(x**2 for x in vector1))
    magnitude2 = math.sqrt(sum(x**2 for x in vector2))
    
    angle_radians = math.acos(dot_product / (magnitude1 * magnitude2))
    return math.degrees(angle_radians)


def approx_best_fit_quadrilateral(contours) -> list[(int, int)]:
    """
    Fit n-sided polygon to contour
    
    Based on: https://stackoverflow.com/questions/41138000/fit-quadrilateral-tetragon-to-a-blob
    Frustum Optimization To Maximize Object’s Image Area
    https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=1fbd43f3827fffeb76641a9c5ab5b625eb5a75ba
    With minimum corner degrees to converge to a box
    """

    hull = cv2.convexHull(contours)
    hull = np.array(hull).reshape((len(hull), 2))

    # to sympy land
    hull = [sympy.Point(*pt) for pt in hull]

    # run until we cut down to 4 vertices
    while len(hull) > 4:
        best_candidate = None

        # for all edges in hull ( <edge_idx_1>, <edge_idx_2> ) ->
        for edge_idx_1 in range(len(hull)):
            edge_idx_2 = (edge_idx_1 + 1) % len(hull)

            adj_idx_1 = (edge_idx_1 - 1) % len(hull)
            adj_idx_2 = (edge_idx_1 + 2) % len(hull)

            edge_pt_1 = sympy.Point(*hull[edge_idx_1])
            edge_pt_2 = sympy.Point(*hull[edge_idx_2])
            adj_pt_1 = sympy.Point(*hull[adj_idx_1])
            adj_pt_2 = sympy.Point(*hull[adj_idx_2])

            subpoly = sympy.Polygon(adj_pt_1, edge_pt_1, edge_pt_2, adj_pt_2)
            angle1 = subpoly.angles[edge_pt_1]
            angle2 = subpoly.angles[edge_pt_2]

            # we need to first make sure that the sum of the interior angles the edge
            # makes with the two adjacent edges is more than 180°
            if sympy.N(angle1 + angle2) <= sympy.pi:
                continue
            
            # Enforce minimum corner degrees
            # if sympy.N(angle1) < min_corner_radians or sympy.N(angle2) < min_corner_radians:
            #     continue            

            # find the new vertex if we delete this edge
            adj_edge_1 = sympy.Line(adj_pt_1, edge_pt_1)
            adj_edge_2 = sympy.Line(edge_pt_2, adj_pt_2)
            intersect = adj_edge_1.intersection(adj_edge_2)[0]

            # the area of the triangle we'll be adding
            area = sympy.N(sympy.Triangle(edge_pt_1, intersect, edge_pt_2).area)
            # should be the lowest
            if best_candidate and best_candidate[1] < area:
                continue

            # delete the edge and add the intersection of adjacent edges to the hull
            better_hull = list(hull)
            better_hull[edge_idx_1] = intersect
            del better_hull[edge_idx_2]
            best_candidate = (better_hull, area)

        if not best_candidate:
            raise ValueError("Could not find the best fit n-gon!")

        hull = best_candidate[0]

    return [Point(x, y) for x, y in hull]

#####################
#       LINE        #
#####################

def extend_line(line: LineString, extension_length=IMAGE_BASE_WIDTH):
    """
    Extend a line by extension length in both directions
    """
    start_point, end_point = line.coords
    # Calculate the direction vector of the line
    direction_vector = np.array(end_point) - np.array(start_point)    
    # Length of the direction vector
    length = np.linalg.norm(direction_vector)    
    # Normalize the direction vector
    normalized_direction = direction_vector / length    
    # Calculate the new start and end points after extension
    extended_start_point = np.array(start_point) - normalized_direction * extension_length
    extended_end_point = np.array(end_point) + normalized_direction * extension_length    
    return LineString([extended_start_point, extended_end_point])


def calculate_line_slope_intercept(line: LineString):
    """
    Calculate the slope and intercept of a line - y = mx + b
    """
    # Extract coordinates of the two points
    p1, p2 = line.coords    
    x1, y1 = p1
    x2, y2 = p2    
    # Calculate the differences in y and x coordinates
    delta_y = y2 - y1
    delta_x = x2 - x1    
    # Calculate the slope
    m = delta_y / delta_x    
    # Calculate the intercept
    b = y1 - m * x1
    return m, b


def calculate_perpendicular_distance_point_to_line(x, y, m, b):   
    """
    Calculate the perpendicular distance from the point to the line
    """
    return abs(m * x - y + b) / np.sqrt(m ** 2 + 1)    


def get_furthest_point_perpendicular_from_line(line: LineString, points):    
    """
    Get the furthest point from a line, calculated at right angles along the length of the line 
    """
    m, b = calculate_line_slope_intercept(line)   
    distances = [calculate_perpendicular_distance_point_to_line(x,y,m,b) for x, y in points]
    return Point(*points[np.argmax(distances)])


def order_points(points: List[Point], clockwise=False):
    """
    Order points - get more reliable results than imutils perspective.order_points 
    """
    hull = cv2.convexHull(points_to_numpy(points), clockwise=clockwise)
    # And back to list of Points
    return [Point(*point.ravel()) for point in hull]

def calculate_line_intersecting_point(line, point: Point, width=2000):
    """
    Calculate new line with the same slope, interceptiing point
    """
    x, y = point
    m, b = calculate_line_slope_intercept(line)
    # Calculate the new intercept for the line passing through x, y
    new_b = y - m * x
    calc_y = lambda x: (round(m * x + new_b))
    point1 = (0, calc_y(0))
    point2 = (width, calc_y(width))
    return LineString([point1, point2])

def points_to_numpy(points: List[Point]) -> np.array:
    """
    Convert list of Points to numpy array
    """
    return np.array([point.to_numpy() for point in points])   