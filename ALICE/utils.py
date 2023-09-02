from hashlib import blake2b
import numpy as np
from copy import deepcopy
from scipy.interpolate import interp1d
from skimage import measure
import cv2
from collections import Counter
from shapely import LineString
from itertools import zip_longest
import sympy
import math


def pairwise(iterable):
    return zip_longest(iterable, iterable[1:], fillvalue=iterable[0])

def iter_list_from_value(lst, value):
    # Iterate through list, starting from value, and looping to start to end at value
    i = lst.index(value)
    yield from lst[i:] + lst[:i]
    

#############################
#       quadrilaterals      #
#############################


def approx_quadrilateral_from_closest_edges(edges, mask):
    """
    Approximate the quadrilateral vertices, from the two closest edges
    
    We create a line with same slope, and then calculate the position intersecting
    the mask at the greatest perpendicular distance from the initial edge
    
    Intersection of these four lines will be the quadrilateral vertices
    
    """
    lines = []
    # Loop through the two good edges, creating a line with the same slope
    # which interesect the further perpendicular edge of the mask
    for edge in edges:
        extended_edge = extend_line(edge, mask.width * mask.height)
        furthest_point = get_furthest_point_perpendicular_from_line(extended_edge, mask.edge_points())
        new_line = get_line_at_point(extended_edge, furthest_point, mask.width)
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


def approx_best_fit_ngon(contours, n: int = 4) -> list[(int, int)]:
    """
    Fit n-sided polygon to contour
    
    From: https://stackoverflow.com/questions/41138000/fit-quadrilateral-tetragon-to-a-blob
    Based on: View Frustum Optimization To Maximize Object’s Image Area
    https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=1fbd43f3827fffeb76641a9c5ab5b625eb5a75ba
    """

    hull = cv2.convexHull(contours)
    hull = np.array(hull).reshape((len(hull), 2))

    # to sympy land
    hull = [sympy.Point(*pt) for pt in hull]

    # run until we cut down to n vertices
    while len(hull) > n:
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

    # back to python land
    hull = [(int(x), int(y)) for x, y in hull]

    return hull

#####################
#       LINE        #
#####################

def extend_line(line: LineString, extension_length):
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
    return points[np.argmax(distances)]


def get_line_at_point(line: LineString, point, image_width):     
    """
    Reposition line with the same slope, so intercepts points
    """
    m, _ = calculate_line_slope_intercept(line) 
    # Calculate new intercept
    b = point[1] - m * point[0]
    return LineString([(0, b), (image_width, image_width * m + b)])

def calculate_angle_between_lines(line1, line2):
    """
    Calculate the angle between lines - will work even if the lines dont't touch    
    """
    start1, end1 = line1.coords
    start2, end2 = line2.coords
    
    # Calculate direction vectors
    vector1 = (end1[0] - start1[0], end1[1] - start1[1])
    vector2 = (end2[0] - start2[0], end2[1] - start2[1])
    
    # Calculate dot product and magnitudes
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    magnitude1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
    magnitude2 = math.sqrt(vector2[0]**2 + vector2[1]**2)
    
    # Calculate cosine of the angle
    cosine_angle = dot_product / (magnitude1 * magnitude2)
    
    # Calculate angle in radians
    angle_radians = math.acos(cosine_angle) 
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees