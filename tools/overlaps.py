import os
import numpy as np
import psutil
import yaml
import open3d as o3d
import copy
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from tools.fileloader import load_files, load_poses, read_pc


def threads_count():
    return psutil.cpu_count()


def remove_outliers(pc, params):
    points = np.asarray(pc.points)
    distances = np.linalg.norm(points, axis=1)
    pc.points = o3d.utility.Vector3dVector(points[(distances >= params['lidar']['eps']) &
                                                  (distances <= params['lidar']['max_range_0.8'])])
    # print(f'Removed {len(points) - len(pc.points)} outliers.')
    return pc


def align_points(scan, pose):
    scan_pose = np.vstack((pose, [0.0, 0.0, 0.0, 1.0]))
    scan_map = scan.transform(scan_pose)
    return scan_map


def kd_tree(pc):
    pc_tree = o3d.geometry.KDTreeFlann(pc)
    return pc_tree


def search_nearest_points(pc_tree, point, num_neighbors):
    [k, idx, dists] = pc_tree.search_knn_vector_3d(point, num_neighbors)   # k: numbers, idx: indices, dists: distances
    return k, idx, dists


def search_radius_points(pc_tree, point, radius):
    [k, idx, dists] = pc_tree.search_radius_vector_3d(point, radius)
    return k, idx, dists


def calculate_overlap(scan1_kd_tree, num_scan1_pts, scan2, params):
    """
    Compute the overlap between two 2 point clouds.

    Args:
        scan1_kd_tree: (o3d.geometry.KDTreeFlann) kd tree if current point cloud.
        num_scan1_pts: (int) number of points in scan1_kd_tree.
        scan2: (o3d.geometry.PointCloud) query point cloud scan.
        params: (dict) parameters for lidar and odom.
    """
    scan2_pts = scan2.points
    num_scan2_pts = len(scan2_pts)
    num_correspondences = 0

    # count number of overlap points
    for i in range(num_scan2_pts):
        k, idx, dist = search_nearest_points(scan1_kd_tree, scan2_pts[i], 1)
        if dist[0] ** 2 <= params['odom']['maxCorrespondenceDistance']:
            num_correspondences += 1

    overlap_pts = num_correspondences
    total_pts = num_scan1_pts + num_scan2_pts - overlap_pts
    intersection_over_union = overlap_pts / total_pts
    return intersection_over_union


def calculate_overlaps_preprocess(poses, pcd_files_path, params):
    point_clouds = []
    point_clouds_kd = []
    for i in tqdm(range(len(pcd_files_path))):
        pose = poses[i]
        pc = read_pc(pcd_files_path[i], format='o3d')
        pc = remove_outliers(pc, params)
        pc = align_points(pc, pose)
        pc_tree = kd_tree(pc)

        point_clouds.append(pc)
        point_clouds_kd.append(pc_tree)

    return point_clouds, point_clouds_kd


def calculate_overlaps_matrix(pcs, kd_trees, params):
    overlaps_matrix = np.zeros((len(pcs), len(pcs)))
    for i in tqdm(range(len(pcs))):
        kd_tree_i = kd_trees[i]
        num_pts_i = len(pcs[i].points)

        for j in range(i, len(pcd_files_path)):
            pc_j = pcs[j]
            overlap = calculate_overlap(kd_tree_i, num_pts_i, pc_j, params)
            overlaps_matrix[i, j] = overlap
            overlaps_matrix[j, i] = overlap

    return overlaps_matrix


if __name__ == '__main__':
    config_path = '/home/vectr/PycharmProjects/lidar_learning/configs/config.yml'
    params_path = '/home/vectr/PycharmProjects/lidar_learning/configs/parameters.yml'

    config = yaml.safe_load(open(config_path))
    params = yaml.safe_load(open(params_path))

    seq = 'sculpture_garden'
    poses_path = os.path.join(config['data_root']['poses'], seq, 'poses.txt')
    pcd_folder_path = os.path.join(config['data_root']['pcd_files'], seq)

    poses = load_poses(poses_path)
    pcd_files_path = load_files(pcd_folder_path)

    # scan1 = 910
    # scan2 = 1660
    # scan3 = 1650
    #
    # # in format of o3d point cloud
    # pc1 = read_pc(pcd_files_path[scan1], format='o3d')
    # pc2 = read_pc(pcd_files_path[scan2], format='o3d')
    # pc3 = read_pc(pcd_files_path[scan3], format='o3d')
    #
    # pose1 = poses[scan1]
    # pose2 = poses[scan2]
    # pose3 = poses[scan3]
    #
    # # copy of raw point cloud for visualization
    # pc1_copy = copy.deepcopy(pc1)
    # pc1_copy.paint_uniform_color([1, 0, 0])
    #
    # # remove outliers
    # pc1 = remove_outliers(pc1, params)
    # pc2 = remove_outliers(pc2, params)
    # pc3 = remove_outliers(pc3, params)
    # o3d.visualization.draw_geometries([pc1_copy, pc1])
    #
    # # alignment scans in same coordinate
    # pc1 = align_points(pc1, pose1)
    # pc2 = align_points(pc2, pose2)
    # pc3 = align_points(pc3, pose3)
    #
    # pc1.paint_uniform_color([1, 0, 0])
    # pc2.paint_uniform_color([0, 1, 0])
    # pc3.paint_uniform_color([0, 0, 1])
    # # o3d.visualization.draw_geometries([pc2, pc1])
    # # o3d.visualization.draw_geometries([pc2, pc3])
    #
    # # calculate overlap between 2 point clouds
    # calculate_overlap(pc1, pc2, params)
    # calculate_overlap(pc2, pc3, params)

    # temp
    pcs, kd_trees = calculate_overlaps_preprocess(poses, pcd_files_path, params)
    overlaps_matrix = calculate_overlaps_matrix(pcs, kd_trees, params)
    np.save(os.path.join(config['data_root']['gt_overlap'], 'overlaps_matrix.npy'), overlaps_matrix)

    # calculate_overlap_preprocess(poses, pcd_files_path, params)

