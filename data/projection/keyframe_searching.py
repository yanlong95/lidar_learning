"""
A temporary file for keyframe image searching based on positions. The keyframes published by DLIO are transformed and
voxelized. Thus, we cannot use keyframes published by DLIO directly. Therefore, we search the keyframes based on their
positions directly, and select the point cloud from full scans directly.
"""
import os
import yaml
import shutil
import faiss
import matplotlib.pyplot as plt

from tools.fileloader import load_xyz_rot, load_files


def indices_searching(poses, poses_kf, src_files, dst_folder):
    """
    Search the indices of keyframes by choosing the closest indices from full scans.

    Args:
        poses: (numpy.array) positions of full scans.
        poses_kf: (numpy.array) positions of keyframe scans.
        src_files: (list) paths of full pcd scans.
        dst_folder: (string) path to save scans for keyframes.
    """
    # searching the indices for keyframes
    index = faiss.IndexFlatL2(3)
    index.add(poses)
    _, indices = index.search(poses_kf, 1)
    indices = indices.squeeze()

    # copy and rename pcd files
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    for i, idx in enumerate(indices):
        # copy files
        src_file = src_files[idx]
        shutil.copy(src_file, dst_folder)

        # rename the file in order number
        src_file_name = os.path.split(src_file)[1]
        os.rename(os.path.join(dst_folder, src_file_name), os.path.join(dst_folder, f'{str(i).zfill(6)}.pcd'))


if __name__ == '__main__':
    config_path = '/home/vectr/PycharmProjects/lidar_learning/configs/config.yml'
    config = yaml.safe_load(open(config_path))
    poses_path = config['data_root']['poses']
    keyframes_path = config['data_root']['keyframes']
    pcd_files_path = config['data_root']['pcd_files']
    seq = config['seqs']['all'][9]

    poses_path_seq = os.path.join(poses_path, seq, 'poses.txt')
    keyframes_poses_path_seq = os.path.join(keyframes_path, seq, 'poses/poses_kf.txt')
    pcd_files_path_seq = os.path.join(pcd_files_path, seq)

    xyz, _ = load_xyz_rot(poses_path_seq)
    xyz_kf, _ = load_xyz_rot(keyframes_poses_path_seq)
    pcd_files = load_files(pcd_files_path_seq)
    dst_folder = os.path.join(keyframes_path, seq, 'pcd_files')

    # searching and copy
    indices_searching(xyz, xyz_kf, pcd_files, dst_folder)
