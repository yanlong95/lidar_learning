import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from typing import Tuple, List, Optional, Union, Any


def compute_nearest_neighbors(
        reference_points: np.ndarray,
        query_points: np.ndarray,
        num_neighbors: int = 1,
        neighbors_indices: bool = True
):
    kd_tree = cKDTree(reference_points)
    distances, indices = kd_tree.query(query_points, k=num_neighbors)
    if neighbors_indices:
        return distances, indices
    else:
        return distances

def compute_overlap(
        scan1: np.ndarray,
        scan2: np.ndarray,
        distance_threshold: float = 0.5,
        num_neighbors: int = 1
):
    """
    Compute the overlap between two point clouds.

    Args:
        scan1: (np.ndarray) reference point cloud (kdtree) in shape (n, 3).
        scan2: (np.ndarray) query point cloud in shape (m, 3).
        distance_threshold: (float) maximum distance between two points to be considered as overlap.
        num_neighbors: (int) number of closest neighbors to be considered.

    Returns:
        overlap: (float) overlap between two point clouds.
        indices: (np.ndarray) indices of the corresponding points in scan1 (reference scan).
        mask_query: (np.ndarray) if scan2 (query scan) has any matching point within threshold.
        distances: (np.ndarray) distances between the corresponding points in scan1 and scan2.
    """
    distances, indices = compute_nearest_neighbors(scan1, scan2, num_neighbors=num_neighbors)
    mask_query = distances <= distance_threshold
    overlap = np.mean(mask_query)
    return overlap, indices, mask_query, distances


if __name__ == '__main__':
    scan1 = np.random.rand(10000, 3) * 50
    scan2 = np.random.rand(10000, 3) * 50
    overlap, _, _, _ = compute_overlap(scan1, scan2)
    print(overlap)
