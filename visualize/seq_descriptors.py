# temp file to test sequential descriptors

import os
import yaml
import numpy as np

from tools.fileloader import load_xyz_rot, load_descriptors


# concatenate n descriptors together to create a sequential descriptor
def stack_descriptors(descriptors, n=5):
    pass


if __name__ == '__main__':
    config_path = '../configs/plot_temp.yaml'
    config = yaml.safe_load(open(config_path))

    seq = 'geo_loop'
    poses_folder_path = config['data_root']['poses']
    descriptors_folder_path = config['data_root']['descriptors']
    keyframes_folder_path = config['data_root']['keyframes']

    poses_path = os.path.join(poses_folder_path, seq, 'poses.txt')
    keyframe_poses_path = os.path.join(keyframes_folder_path, seq, 'poses/poses_kf.txt')

    descriptors_path = os.path.join(descriptors_folder_path, seq, 'test_whole_frame_descriptors.npy')
    keyframe_descriptors_path = os.path.join(descriptors_folder_path, seq, 'keyframe_descriptors.npy')

    xyz, _ = load_xyz_rot(poses_path)
    xyz_kf, _ = load_xyz_rot(keyframe_poses_path)

    descriptors = load_descriptors(descriptors_path)
    descriptors_kf = load_descriptors(keyframe_descriptors_path)
