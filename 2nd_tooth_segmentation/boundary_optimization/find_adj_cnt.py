from scipy.spatial import cKDTree
import numpy as np

def find_nearest_points(points, X):

    points = np.array(points)
    points = points.transpose(1,0)
    points = points.T

    tree = cKDTree(points)
    distances, indices = tree.query(X, k=7)
    nearest_points = points[indices] 

    return nearest_points

