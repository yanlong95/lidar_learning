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
    def __init__(self, pc_remove_outliers=False, pc_align=True, min_range=-1.0, max_range=10000.0,
                 max_correspondence_distance=0.5):
        self.pc_remove_outliers = pc_remove_outliers
        self.pc_align = pc_align
        self.min_range = min_range
        self.max_range = max_range
        self.max_correspondence_distance = max_correspondence_distance

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

    def calculate_overlap(self, scan1_kd_tree, num_scan1_pts, scan2, max_correspondence_distance=0.5, num_neighbors=100):
        """
        Compute the overlap between two 2 point clouds.

        Args:
            scan1_kd_tree: (o3d.geometry.KDTreeFlann) kd tree of current point cloud.
            num_scan1_pts: (int) number of points in scan1_kd_tree.
            scan2: (o3d.geometry.PointCloud) query point cloud scan.
            max_correspondence_distance: (double) maximum distance between 2 points to be considered as correspondence.
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
            if dist[0] <= max_correspondence_distance ** 2:
                num_correspondences += 1
            indices_correspondences[i, :] = np.asarray(idx)
            dists_correspondences[i, :] = np.asarray(dist) ** 0.5

        # overlap define as intersection over union of 2 point clouds
        overlap_pts = num_correspondences
        total_pts = num_scan1_pts + num_scan2_pts - overlap_pts
        intersection_over_union = overlap_pts / total_pts
        masks_correspondences = dists_correspondences[:, 0] <= max_correspondence_distance
        return intersection_over_union, indices_correspondences, dists_correspondences, masks_correspondences

    def do_matching(self, pointcloud1, pointcloud2, pose1=None, pose2=None, num_neighbors=100):
        """
        Find the matching points between two point clouds.

        Args:
            pointcloud1: (numpy.ndarray or o3d.geometry.PointCloud) point cloud 1.
            pointcloud2: (numpy.ndarray or o3d.geometry.PointCloud) point cloud 2.
            pose1: (numpy.ndarray) pose of point cloud 1. If None, no alignment is performed.
            pose2: (numpy.ndarray) pose of point cloud 2. If None, no alignment is performed.
            num_neighbors: (int) the number of neighbors to search, default is 100.
        """
        if isinstance(pointcloud1, np.ndarray):
            pc1 = o3d.geometry.PointCloud()
            pc1.points = o3d.utility.Vector3dVector(pointcloud1)
        elif isinstance(pointcloud1, o3d.geometry.PointCloud):
            pc1 = pointcloud1
        else:
            raise ValueError('Invalid point cloud type (only support numpy and o3d.geometry.PointCloud).')

        if isinstance(pointcloud2, np.ndarray):
            pc2 = o3d.geometry.PointCloud()
            pc2.points = o3d.utility.Vector3dVector(pointcloud2)
        elif isinstance(pointcloud2, o3d.geometry.PointCloud):
            pc2 = pointcloud2
        else:
            raise ValueError('Invalid point cloud type (only support numpy and o3d.geometry.PointCloud).')

        if self.pc_remove_outliers:
            pc1 = self.remove_outliers(pc1, self.min_range, self.max_range)
            pc2 = self.remove_outliers(pc2, self.min_range, self.max_range)

        if self.pc_align and pose1 is not None:
            pc1 = self.align_points(pc1, pose1)
        if self.pc_align and pose2 is not None:
            pc2 = self.align_points(pc2, pose2)

        pc1_tree = self.kd_tree(pc1)
        overlap, indices, dists, masks = self.calculate_overlap(pc1_tree, len(pc1.points), pc2,
                                                                self.max_correspondence_distance, num_neighbors)
        return overlap, indices, dists, masks

    def do_matching_indices(self, pointcloud1, pointcloud2, pose1=None, pose2=None):
        """
        Find the matching indices between two point clouds.

        Args:
            pointcloud1: (numpy.ndarray or o3d.geometry.PointCloud) point cloud 1.
            pointcloud2: (numpy.ndarray or o3d.geometry.PointCloud) point cloud 2.
            pose1: (numpy.ndarray) pose of point cloud 1. If None, no alignment is performed.
            pose2: (numpy.ndarray) pose of point cloud 2. If None, no alignment is performed.
        Returns:
            pc1_indices: (numpy.ndarray) indices of point cloud 1, in shape (n, ).
            pc2_indices: (numpy.ndarray) indices of point cloud 2, in shape (n, ).
        """
        _, indices, _, masks = self.do_matching(pointcloud1, pointcloud2, pose1, pose2, num_neighbors=1)
        pc1_indices = np.arange(len(indices))[masks]
        pc2_indices = indices[masks, 0]
        return pc1_indices, pc2_indices

    def matching_chamfer(self, pointcloud1, pointcloud2, pose1, pose2):
        """
        Compute the chamfer overlap between two point clouds.

        Args:
            pointcloud1: (numpy.ndarray or o3d.geometry.Pointcloud) point cloud 1.
            pointcloud2: (numpy.ndarray or o3d.geometry.Pointcloud) point cloud 2.
            pose1: (numpy.ndarray) pose of point cloud 1.
            pose2: (numpy.ndarray) pose of point cloud 2.
        """
        overlap1, indices1, dists1, masks1 = self.do_matching(pointcloud1, pointcloud2, pose1, pose2)
        overlap2, indices2, dists2, masks2 = self.do_matching(pointcloud2, pointcloud1, pose2, pose1)

        print(overlap1)
        print(overlap2)


if __name__ == '__main__':
    pcd_paths = load_files('/media/vectr/vectr3/Dataset/overlap_transformer/pcd_files/botanical_garden')
    poses = load_poses('/media/vectr/vectr3/Dataset/overlap_transformer/poses/botanical_garden/poses.txt')
    xyz, _ = load_xyz_rot('/media/vectr/vectr3/Dataset/overlap_transformer/poses/botanical_garden/poses.txt')
    overlaps = load_overlaps('/media/vectr/vectr3/Dataset/overlap_transformer/overlaps/botanical_garden.bin')

    idx1 = 450
    idx2 = 691

    pc1 = read_pc(pcd_paths[idx1])
    pc2 = read_pc(pcd_paths[idx2])

    matcher = PointMatcher()
    # overlap, indices, dists, masks = matcher.do_matching(pc1, pc2, poses[idx1], poses[idx2])
    # pc1_indices, pc2_indices = matcher.do_matching_indices(pc1, pc2, poses[idx1], poses[idx2])

    matcher.matching_chamfer(pc1, pc2, poses[idx1], poses[idx2])
