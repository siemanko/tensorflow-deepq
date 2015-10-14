"""
This module assumes that all geometrical points are
represented as 1D numpy arrays.

It was designed and tested on 2D points,
but if you try it on 3D points you may
be pleasantly surprised ;-) 
"""
import numpy as np


def point_distance(x, y):
    """Returns euclidean distance between points x and y"""
    return np.linalg.norm(x-y)

def point_projected_on_line(line_s, line_e, point):
    """Project point on line that goes through line_s and line_e

    assumes line_e is not equal or close to line_s
    """
    line_along = line_e - line_s
    
    transformed_point = point - line_s
    
    point_dot_line  = np.dot(transformed_point, line_along)
    line_along_norm = np.dot(line_along, line_along)
    
    transformed_projection = (point_dot_line / line_along_norm) * line_along
    
    return transformed_projection + line_s

def point_segment_distance(segment_s, segment_e, point):
    """Returns distance from point to the closest point on segment
    connecting points segment_s and segment_e"""
    projected = point_projected_on_line(segment_s, segment_e, point)
    if np.isclose(point_distance(segment_s, projected) + point_distance(projected, segment_e),
        point_distance(segment_s, segment_e)):
        # projected on segment
        return point_distance(point, projected)
    else:
        return min(point_distance(point, segment_s), point_distance(point, segment_e))
