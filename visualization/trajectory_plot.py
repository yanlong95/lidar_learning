import os
import yaml
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tools.fileloader import load_xyz_rot, load_poses

def trajectory2D(poses, poses_ky):
    dists = np.linalg.norm(poses[:, :2], axis=1)
    plt.figure()
    plt.scatter(poses[:, 0], poses[:, 1], c=dists/np.sum(dists), s=1, alpha=0.1, cmap='viridis', label='Trajectory')
    # plt.scatter(poses_ky[:, 0], poses_ky[:, 1], c='red', s=5, label='Keyframes')
    # plt.axis('square')
    plt.xlabel('X [m]', fontsize=14)
    plt.ylabel('Y [m]', fontsize=14)
    plt.title('Trajectory')
    plt.legend()
    plt.show()


def trajectory3D(poses, poses_ky):
    dists = np.linalg.norm(poses, axis=1)
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(poses[:, 0], poses[:, 1], poses[:, 2], c=dists, s=1, alpha=0.1, cmap='viridis', label='Trajectory')
    ax.scatter3D(poses_ky[:, 0], poses_ky[:, 1], poses_ky[:, 2], c='red', s=5, label='Trajectory')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    plt.title('Trajectory')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    config_path = '../configs/plot_temp.yaml'
    config = yaml.safe_load(open(config_path))

    data_root = config['data_root']['root']
    poses_folder_path = config['data_root']['poses']
    descriptors_path = config['data_root']['descriptors']
    keyframes_path = config['data_root']['keyframes']
    seqs = config['seqs']

    seq = seqs[8]
    poses_path = os.path.join(poses_folder_path, seq, 'poses.txt')
    keyframe_poses_path = os.path.join(keyframes_path, seq, 'poses/poses_kf.txt')

    xyz, _ = load_xyz_rot(poses_path)
    xyz_kf, _ = load_xyz_rot(keyframe_poses_path)

    poses_true = load_poses(os.path.join(data_root, f'test/{seq}/predictions/true/poses.npy'))
    poses_false = load_poses(os.path.join(data_root, f'test/{seq}/predictions/false/poses.npy'))

    indices_true = load_poses(os.path.join(data_root, f'test/{seq}/predictions/true/indices.npy'))
    indices_false = load_poses(os.path.join(data_root, f'test/{seq}/predictions/false/indices.npy'))

    # trajectory2D(xyz, xyz_kf)
    trajectory2D(xyz, xyz_kf)
