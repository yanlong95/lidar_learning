import os
import yaml
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from tools.fileloader import load_xyz_rot

def trajectory2D(poses, poses_ky):
    dists = np.linalg.norm(poses[:, :2], axis=1)
    plt.figure()
    plt.scatter(poses[:, 0], poses[:, 1], c=dists/np.sum(dists), s=1, alpha=0.1, cmap='viridis', label='Trajectory')
    plt.scatter(poses_ky[:, 0], poses_ky[:, 1], c='red', s=5, label='Keyframes')
    # plt.axis('square')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('Overlap Map')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    config_path = '../configs/plot_temp.yaml'
    config = yaml.safe_load(open(config_path))

    poses_folder_path = config['data_root']['poses']
    descriptors_path = config['data_root']['descriptors']
    keyframes_path = config['data_root']['keyframes']
    seqs = config['seqs']

    seq = seqs[0]
    poses_path = os.path.join(poses_folder_path, seq, 'poses.txt')
    keyframe_poses_path = os.path.join(keyframes_path, seq, 'poses/poses_kf.txt')

    xyz, _ = load_xyz_rot(poses_path)
    xyz_kf, _ = load_xyz_rot(keyframe_poses_path)

    trajectory2D(xyz, xyz_kf)
