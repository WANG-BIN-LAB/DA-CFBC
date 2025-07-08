import numpy as np
from scipy.spatial import distance
import numpy as np

def process_labels(crop_i, labels, bd, crop_size, nearest_points, patches_coord_min):

    coords = np.argwhere(labels > 0)
    coords_f = coords.astype(float).transpose(1,0)  

    coords_f[0] = coords_f[0] + patches_coord_min[crop_i, 0]
    coords_f[1] = coords_f[1] + patches_coord_min[crop_i, 1]
    coords_f[2] = coords_f[2] + patches_coord_min[crop_i, 2]
    coords_f = coords_f.transpose(1,0)
    distances = distance.cdist(coords_f, nearest_points, 'euclidean')
    min_indices = np.argmin(distances, axis=1)
    points_to_zero = (min_indices != 0)
    count_not_zero = np.sum(points_to_zero)
    updated_labels = np.copy(labels)
    coord_del = coords[points_to_zero].transpose(1,0)
    coord_del1 = coord_del.transpose(1,0)
    bd[bd!=1] = 0
    tooth_id = labels[int(crop_size[0]/2), int(crop_size[1]/2), int(crop_size[2]/2)]
    indices0 = np.where(labels == tooth_id)

    coord = list(zip(indices0[0], indices0[1], indices0[2]))
    coord_true = np.array(coord)
    set1 = set(map(tuple, coord_del1))
    set2 = set(map(tuple, coord_true))
    if set1:
        common_points_set = set1.intersection(set2)
        coord_del = np.array([list(point) for point in set1 - common_points_set]).transpose()
        
        if coord_del.size > 0:  
            updated_labels[coord_del[0], coord_del[1], coord_del[2]] = 0
        else:
            pass

    return updated_labels
