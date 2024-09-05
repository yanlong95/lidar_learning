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


def transform_point_cloud(
        point_cloud: np.ndarray,
        transformation: np.ndarray,
        inverse: bool = False
):
    """
    Transform a point cloud from local frame to global frame.

    Args:
        point_cloud: (np.ndarray) point cloud in shape (n, 3).
        transformation: (np.ndarray) transformation matrix in shape (4, 4) or (3, 4).
        inverse: (bool) if True, use the inverse transformation.

    Returns:
        transformed_point_cloud: (np.ndarray) transformed point cloud in shape (n, 3).
    """
    if transformation.shape == (3, 4):
        transformation = np.vstack((transformation, np.array([0, 0, 0, 1])))

    if inverse:
        transformation = np.linalg.inv(transformation)

    point_cloud = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))
    transformed_point_cloud = np.dot(transformation, point_cloud.T).T
    return transformed_point_cloud[:, :3]


if __name__ == '__main__':
    scan1 = np.random.rand(10000, 3) * 50
    scan2 = np.random.rand(10000, 3) * 50
    overlap, _, _, _ = compute_overlap(scan1, scan2)
    print(overlap)
