import os
import numpy as np
import yaml
import open3d as o3d
from tqdm import tqdm

from tools.fileloader import load_files, load_poses, read_pc


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


def calculate_overlaps_preprocess(poses, pcd_files_path, saving_path, params):
    for i in tqdm(range(len(pcd_files_path))):
        pose = poses[i]
        pc = read_pc(pcd_files_path[i], format='o3d')
        pc = remove_outliers(pc, params)
        pc = align_points(pc, pose)

        if not os.path.exists(saving_path):
            os.makedirs(saving_path)

        o3d.io.write_point_cloud(os.path.join(saving_path, f'{str(i).zfill(6)}.pcd'), pc)


if __name__ == '__main__':
    config_path = '/home/vectr/PycharmProjects/lidar_learning/configs/config.yml'
    params_path = '/home/vectr/PycharmProjects/lidar_learning/configs/parameters.yml'

    config = yaml.safe_load(open(config_path))
    params = yaml.safe_load(open(params_path))

    seq = 'sculpture_garden'
    poses_path = os.path.join(config['data_root']['poses'], seq, 'poses.txt')
    pcd_folder_path = os.path.join(config['data_root']['pcd_files'], seq)
    pcd_filtered_path = os.path.join(config['data_root']['pcd_files'], 'filtered', seq)

    poses = load_poses(poses_path)
    pcd_files_path = load_files(pcd_folder_path)

    # scan1 = 910
    # scan2 = 1660
    # scan3 = 1650

    calculate_overlaps_preprocess(poses, pcd_files_path, pcd_filtered_path, params)

