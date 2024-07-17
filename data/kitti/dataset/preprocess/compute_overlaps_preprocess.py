"""
Preprocess the raw point cloud. Align the point clouds in global position. Save the aligned point clouds in .pcd files.
Note, the original script compute the overlaps directly, but the computing speed in python is slow. Thus, this script
just do the preprocess and save the aligned point clouds. The computation of overlaps is done in C++ in a separate file.
"""
import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
from pathlib import Path

from tools.fileloader import load_files, load_poses, read_pc, save_pc


def align_points(pc, pose):
    """
    Align the point cloud to original point.

    Args:
        pc: (o3d.geometry.PointCloud) o3d point cloud.
        pose: (numpy.array) a (3, 4) numpy array representing the 3 * 3 rotation matrix and 3 * 1 translation vector.
    Returns:
        pc_map: (o3d.geometry.PointCloud) aligned point cloud.
    """
    pc_pose = np.vstack((pose, [0.0, 0.0, 0.0, 1.0]))
    pc_map = pc.transform(pc_pose)
    return pc_map


def calculate_overlaps_preprocess(poses, pcd_files_path, dst_path):
    """
    Preprocess the raw point cloud. Align the point clouds in global position.

    Args:
        poses: (numpy.array) a (3, 4) numpy array representing the 3 * 3 rotation matrix and 3 * 1 translation vector.
        pcd_files_path: (list of string) the paths of pcd files.
        dst_path: (string) the path of directory to save the processed point cloud.
    """
    full_path = os.path.expanduser(dst_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f'File {full_path} does not exist!')

    for i in tqdm(range(len(pcd_files_path))):
        pose = poses[i]
        pc = read_pc(pcd_files_path[i])
        pc_o3d = o3d.t.geometry.PointCloud()
        pc_o3d.point['positions'] = o3d.core.Tensor(pc[:, :3])
        pc = align_points(pc_o3d, pose)

        fn = Path(pcd_files_path[i]).stem
        saving_path = os.path.join(dst_path, f'{fn}.pcd')
        save_pc(pc, saving_path)


if __name__ == '__main__':
    folder_path = '/media/vectr/T7/Datasets/public_datasets/kitti/dataset/sequences'
    sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16',
                 '17', '18', '19', '20', '21']

    for seq in sequences:
        src_path = os.path.join(folder_path, seq)
        src_pcd = load_files(os.path.join(src_path, 'velodyne'))
        src_poses = load_poses(os.path.join(src_path, 'poses', 'poses.txt'))
        dst_path = os.path.join(src_path, 'pcd_files')
        calculate_overlaps_preprocess(src_poses, src_pcd, dst_path)
