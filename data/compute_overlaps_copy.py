"""
File to compute the overlaps between scans of a sequence of point clouds. The overlap is defined as ratio between the
intersection over union of two point clouds (in 3d space).

Note, due to the size of the sequence, the computation speed could be very slow. A separate c++ file is used to compute
the overlaps matrices in real.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import psutil
import yaml
import copy
import open3d as o3d
from tqdm import tqdm

from tools.fileloader import load_files, load_poses, read_pc, load_xyz_rot, load_overlaps


class PointMatcher:
    def __init__(self, min_range=-1, max_range=-1):
        self.min_range = min_range
        self.max_range = max_range

    def remove_outliers(self, pc, min_range=-1.0, max_range=10000.0):
        """
        Remove outliers of a point cloud. The outliers are considered the points outside the maximum range or 0.

        Args:
            pc: (o3d.geometry.PointCloud) o3d point cloud.
            min_range: (float) minimum range of the point cloud. Default -1.
            max_range: (float) maximum range of the point cloud. Default 10000.
        Returns:
            pc: (o3d.geometry.PointCloud) filtered o3d point.
        """
        points = np.asarray(pc.points)
        distances = np.linalg.norm(points[:, :3], axis=1)
        points = points[(distances >= min_range) & (distances <= max_range)]
        pc.points = o3d.utility.Vector3dVector(points)
        return pc

    def align_points(self, pc, pose):
        """
        Align the point cloud to global frame.

        Args:
            pc: (o3d.geometry.PointCloud) o3d point cloud.
            pose: (numpy.array) a (3, 4) numpy array representing the 3 * 3 rotation matrix and 3 * 1 translation vector.
        Returns:
            pc: (o3d.geometry.PointCloud) aligned point cloud.
        """
        if pose.shape[0] == 3:
            pose = np.vstack((pose, [0.0, 0.0, 0.0, 1.0]))

        return pc.transform(pose)

    def kd_tree(self, pc):
        """
        Build a kdtree for a point cloud.

        Args:
            pc: (o3d.geometry.PointCloud) o3d point cloud.
        Returns:
            pc_tree: (o3d.geometry.KDTreeFlann) a KDTree of input point cloud.
        """
        return o3d.geometry.KDTreeFlann(pc)

    def search_nearest_points(self, pc_tree, point, num_neighbors=1):
        """
        Search the closest point(s) of a given point and kdtree.

        Args:
            pc_tree: (o3d.geometry.KDTreeFlann) o3d point cloud kdtree.
            point: (numpy.array) a point (xyz) in numpy array.
            num_neighbors: (int) the number of closest neighbors.
        Returns:
            k: (int) numbers.
            idx: (o3d.utility.IntVector) index of the closest points.
            dists: (o3d.utility.DoubleVector) distances to the closest points.
        """
        [k, idx, dists] = pc_tree.search_knn_vector_3d(point, num_neighbors)   # k: numbers, idx: indices, dists: distances
        return k, idx, dists

    def search_radius_points(self, pc_tree, point, radius):
        """
        Search the points within a radius of a given point and kdtree.

        Args:
            pc_tree: (o3d.geometry.KDTreeFlann) o3d point cloud kdtree.
            point: (numpy.array) a point (xyz) in numpy array.
            radius: (double) radius.
        Returns:
            k: (int) numbers.
            idx: (o3d.utility.IntVector) index of the points in radius.
            dists: (o3d.utility.DoubleVector) distances to the points in radius.
        """
        [k, idx, dists] = pc_tree.search_radius_vector_3d(point, radius)
        return k, idx, dists

    def calculate_overlap(self, scan1_kd_tree, num_scan1_pts, scan2, maxCorrespondenceDistance=0.5, num_neighbors=100):
        """
        Compute the overlap between two 2 point clouds.

        Args:
            scan1_kd_tree: (o3d.geometry.KDTreeFlann) kd tree of current point cloud.
            num_scan1_pts: (int) number of points in scan1_kd_tree.
            scan2: (o3d.geometry.PointCloud) query point cloud scan.
            maxCorrespondenceDistance: (double) maximum distance between 2 points to be considered as correspondence.
            num_neighbors: (int) the number of neighbors to search.
        Returns:
            intersection_over_union: (double) the overlap ratio (0 ~ 1).
        """
        scan2_pts = scan2.points
        num_scan2_pts = len(scan2_pts)
        num_correspondences = 0
        indices_correspondences = np.zeros((num_scan2_pts, num_neighbors), dtype=int)
        dists_correspondences = np.zeros((num_scan2_pts, num_neighbors), dtype=float)

        # count number of overlap points
        for i in range(num_scan2_pts):
            k, idx, dist = self.search_nearest_points(scan1_kd_tree, scan2_pts[i], num_neighbors)
            if dist[0] <= maxCorrespondenceDistance ** 2:
                num_correspondences += 1
            indices_correspondences[i, :] = np.asarray(idx)
            dists_correspondences[i, :] = np.asarray(dist) ** 0.5

        # overlap define as intersection over union of 2 point clouds
        overlap_pts = num_correspondences
        total_pts = num_scan1_pts + num_scan2_pts - overlap_pts
        intersection_over_union = overlap_pts / total_pts
        return intersection_over_union, indices_correspondences, dists_correspondences


if __name__ == '__main__':
    matcher = PointMatcher()

    pcd_paths = load_files('/media/vectr/vectr3/Dataset/overlap_transformer/pcd_files/botanical_garden')
    poses = load_poses('/media/vectr/vectr3/Dataset/overlap_transformer/poses/botanical_garden/poses.txt')
    xyz, _ = load_xyz_rot('/media/vectr/vectr3/Dataset/overlap_transformer/poses/botanical_garden/poses.txt')
    overlaps = load_overlaps('/media/vectr/vectr3/Dataset/overlap_transformer/overlaps/botanical_garden.bin')

    idx1 = 450
    idx2 = 661

    pc1 = read_pc(pcd_paths[idx1])
    pc1_copy = copy.deepcopy(pc1)
    pc2 = read_pc(pcd_paths[idx2])
    pc2_copy = copy.deepcopy(pc2)

    pc1 = o3d.geometry.PointCloud()
    pc1.points = o3d.utility.Vector3dVector(pc1_copy[:, :3])
    pc2 = o3d.geometry.PointCloud()
    pc2.points = o3d.utility.Vector3dVector(pc2_copy[:, :3])

    pc1 = matcher.align_points(pc1, poses[idx1])
    pc2 = matcher.align_points(pc2, poses[idx2])

    pc1_tree = matcher.kd_tree(pc1)

    o, indices, dists = matcher.calculate_overlap(pc1_tree, len(pc1.points), pc2, maxCorrespondenceDistance=0.5, num_neighbors=100)
    print(o)
    print(overlaps[idx1, idx2])



#
# def remove_outliers(pc, params):
#     """
#     Remove outliers of a point cloud. The outliers are considered the points outside the maximum range or 0.
#
#     Args:
#         pc: (o3d.geometry.PointCloud) o3d point cloud.
#         params: (dict) parameters for lidar and odom.
#     Returns:
#         pc: (o3d.geometry.PointCloud) filtered o3d point.
#     """
#     points = np.asarray(pc.points)
#     distances = np.linalg.norm(points, axis=1)
#     pc.points = o3d.utility.Vector3dVector(points[(distances >= params['lidar']['eps']) &
#                                                   (distances <= params['lidar']['max_range_0.8'])])
#     # print(f'Removed {len(points) - len(pc.points)} outliers.')
#     return pc
#
#
# def align_points(pc, pose):
#     """
#     Align the point cloud to original point.
#
#     Args:
#         pc: (o3d.geometry.PointCloud) o3d point cloud.
#         pose: (numpy.array) a (3, 4) numpy array representing the 3 * 3 rotation matrix and 3 * 1 translation vector.
#     Returns:
#         pc_map: (o3d.geometry.PointCloud) aligned point cloud.
#     """
#     pc_pose = np.vstack((pose, [0.0, 0.0, 0.0, 1.0]))
#     pc_map = pc.transform(pc_pose)
#     return pc_map
#
#
# def kd_tree(pc):
#     """
#     Build a kdtree for a point cloud.
#
#     Args:
#         pc: (o3d.geometry.PointCloud) o3d point cloud.
#     Returns:
#         pc_tree: (o3d.geometry.KDTreeFlann) a KDTree of input point cloud.
#     """
#     pc_tree = o3d.geometry.KDTreeFlann(pc)
#     return pc_tree
#
#
# def search_nearest_points(pc_tree, point, num_neighbors=1):
#     """
#     Search the closest point(s) of a given point and kdtree.
#
#     Args:
#         pc_tree: (o3d.geometry.KDTreeFlann) o3d point cloud kdtree.
#         point: (numpy.array) a point (xyz) in numpy array.
#         num_neighbors: (int) the number of closest neighbors.
#     Returns:
#         k: (int) numbers.
#         idx: (o3d.utility.IntVector) index of the closest points.
#         dists: (o3d.utility.DoubleVector) distances to the closest points.
#     """
#     [k, idx, dists] = pc_tree.search_knn_vector_3d(point, num_neighbors)   # k: numbers, idx: indices, dists: distances
#     return k, idx, dists
#
#
# def search_radius_points(pc_tree, point, radius):
#     """
#     Search the points within a radius of a given point and kdtree.
#
#     Args:
#         pc_tree: (o3d.geometry.KDTreeFlann) o3d point cloud kdtree.
#         point: (numpy.array) a point (xyz) in numpy array.
#         radius: (double) radius.
#     Returns:
#         k: (int) numbers.
#         idx: (o3d.utility.IntVector) index of the points in radius.
#         dists: (o3d.utility.DoubleVector) distances to the points in radius.
#     """
#     [k, idx, dists] = pc_tree.search_radius_vector_3d(point, radius)
#     return k, idx, dists
#
#
# def calculate_overlap(scan1_kd_tree, num_scan1_pts, scan2, params):
#     """
#     Compute the overlap between two 2 point clouds.
#
#     Args:
#         scan1_kd_tree: (o3d.geometry.KDTreeFlann) kd tree if current point cloud.
#         num_scan1_pts: (int) number of points in scan1_kd_tree.
#         scan2: (o3d.geometry.PointCloud) query point cloud scan.
#         params: (dict) parameters for lidar and odom.
#     Returns:
#         intersection_over_union: (double) the overlap ratio (0 ~ 1).
#     """
#     scan2_pts = scan2.points
#     num_scan2_pts = len(scan2_pts)
#     num_correspondences = 0
#
#     # count number of overlap points
#     for i in range(num_scan2_pts):
#         k, idx, dist = search_nearest_points(scan1_kd_tree, scan2_pts[i], 1)
#         if dist[0] ** 2 <= params['odom']['maxCorrespondenceDistance']:
#             num_correspondences += 1
#
#     # overlap define as intersection over union of 2 point clouds
#     overlap_pts = num_correspondences
#     total_pts = num_scan1_pts + num_scan2_pts - overlap_pts
#     intersection_over_union = overlap_pts / total_pts
#     return intersection_over_union
#
#
# def calculate_overlaps_preprocess(poses, pcd_files_path, params):
#     """
#     Preprocess the raw point cloud. Remove the outliers -> align the pose -> build and save kdtree and new point cloud.
#
#     Args:
#         pose: (numpy.array) a (3, 4) numpy array representing the 3 * 3 rotation matrix and 3 * 1 translation vector.
#         pcd_files_path: (list of string) the paths of pcd files.
#         params: (dict) parameters for lidar and odom.
#     Returns:
#         point_clouds: (list of o3d.geometry.PointCloud) list of filtered point clouds.
#         point_clouds_kd: (list of o3d.geometry.KDTreeFlann) list of kd tree.
#     """
#     point_clouds = []
#     point_clouds_kd = []
#     for i in tqdm(range(len(pcd_files_path))):
#         pose = poses[i]
#         pc = read_pc(pcd_files_path[i], format='o3d')
#         pc = remove_outliers(pc, params)
#         pc = align_points(pc, pose)
#         pc_tree = kd_tree(pc)
#
#         point_clouds.append(pc)
#         point_clouds_kd.append(pc_tree)
#
#     return point_clouds, point_clouds_kd
#
#
# def calculate_overlaps_matrix(pcs, kd_trees, params):
#     """
#     Calculate the overlaps between all the scans and save the results in a symmetric matrix.
#
#     Args:
#         pcs: (list of o3d.geometry.PointCloud) the list of o3d point cloud.
#         kd_trees: (list of o3d.geometry.KDTreeFlann) the list of kd trees of pcs.
#         params: (dict) parameters for lidar and odom.
#     Returns:
#         overlaps_matrix: (numpy.array) the overlap matrix for each pair of point clouds.
#     """
#     overlaps_matrix = np.zeros((len(pcs), len(pcs)))
#     for i in tqdm(range(len(pcs))):
#         kd_tree_i = kd_trees[i]
#         num_pts_i = len(pcs[i].points)
#
#         for j in range(i, len(pcd_files_path)):
#             pc_j = pcs[j]
#             overlap = calculate_overlap(kd_tree_i, num_pts_i, pc_j, params)
#             overlaps_matrix[i, j] = overlap
#             overlaps_matrix[j, i] = overlap
#
#     return overlaps_matrix
#
#
# if __name__ == '__main__':
#     # load configuration and parameters
#     config_path = '/configs/config.yml'
#     params_path = '/configs/parameters.yml'
#
#     config = yaml.safe_load(open(config_path))
#     params = yaml.safe_load(open(params_path))
#
#     # load poses and pcd files
#     seq = 'bomb_shelter'
#     poses_path = os.path.join(config['data_root']['poses'], seq, 'poses.txt')
#     pcd_folder_path = os.path.join(config['data_root']['pcd_files'], seq)
#
#     poses = load_poses(poses_path)
#     pcd_files_path = load_files(pcd_folder_path)
#
#     # remove outliers, align point clouds, build kdtrees
#     pcs, kd_trees = calculate_overlaps_preprocess(poses, pcd_files_path, params)
#
#     # calculate the overlaps
#     overlaps_matrix = calculate_overlaps_matrix(pcs, kd_trees, params)
#
#     # save overlaps result in a numpy array
#     np.save(os.path.join(config['data_root']['gt_overlap'], 'overlaps_matrix.npy'), overlaps_matrix)
