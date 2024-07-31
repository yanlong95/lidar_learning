import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

from tools.fileloader import load_files, read_pc, load_xyz_rot
from data.projection.range_image import range_projection


def rot_pc(pc_loc, rot_loc, xyz_loc, rot_ref, xyz_ref):
    pc_world = pc_loc @ rot_loc.T + xyz_loc
    pc_ref = pc_world @ rot_ref - xyz_ref @ rot_ref
    return pc_ref


if __name__ == '__main__':
    config_path = '/home/vectr/PycharmProjects/lidar_learning/configs/config.yml'
    params_path = '/home/vectr/PycharmProjects/lidar_learning/configs/parameters.yml'

    config = yaml.safe_load(open(config_path))
    params = yaml.safe_load(open(params_path))

    sequence = 'botanical_garden'

    submap_path = f'/media/vectr/vectr3/Dataset/overlap_transformer/submaps/overlap_heuristic/{sequence}/pos_neg.npy'
    submap = np.load(submap_path)

    idx = 1000
    keyframes = submap[idx, :]
    print(f'Keyframes indices: {keyframes}')

    pc_paths = load_files(os.path.join(config['data_root']['pcd_files'], sequence))

    pc_ = read_pc(pc_paths[idx])
    pc_kf0 = read_pc(pc_paths[keyframes[0]])
    pc_kf1 = read_pc(pc_paths[keyframes[1]])
    pc_kf2 = read_pc(pc_paths[keyframes[2]])
    pc_kf3 = read_pc(pc_paths[keyframes[3]])
    pc_kf4 = read_pc(pc_paths[keyframes[4]])

    xyz, rot = load_xyz_rot(os.path.join(config['data_root']['poses'], sequence, 'poses.txt'))
    xyz_ = xyz[idx]
    rot_ = rot[idx]
    xyz_kf0 = xyz[keyframes[0]]
    rot_kf0 = rot[keyframes[0]]
    xyz_kf1 = xyz[keyframes[1]]
    rot_kf1 = rot[keyframes[1]]
    xyz_kf2 = xyz[keyframes[2]]
    rot_kf2 = rot[keyframes[2]]
    xyz_kf3 = xyz[keyframes[3]]
    rot_kf3 = rot[keyframes[3]]
    xyz_kf4 = xyz[keyframes[4]]
    rot_kf4 = rot[keyframes[4]]

    pc_kf0_local = rot_pc(pc_kf0, rot_kf0, xyz_kf0, rot_, xyz_)
    pc_kf1_local = rot_pc(pc_kf1, rot_kf1, xyz_kf1, rot_, xyz_)
    pc_kf2_local = rot_pc(pc_kf2, rot_kf2, xyz_kf2, rot_, xyz_)
    pc_kf3_local = rot_pc(pc_kf3, rot_kf3, xyz_kf3, rot_, xyz_)
    pc_kf4_local = rot_pc(pc_kf4, rot_kf4, xyz_kf4, rot_, xyz_)

    img_ = range_projection(pc_, params['lidar'], 32, 512)
    img_kf0 = range_projection(pc_kf0_local, params['lidar'], 32, 512)
    img_kf1 = range_projection(pc_kf1_local, params['lidar'], 32, 512)
    img_kf2 = range_projection(pc_kf2_local, params['lidar'], 32, 512)
    img_kf3 = range_projection(pc_kf3_local, params['lidar'], 32, 512)
    img_kf4 = range_projection(pc_kf4_local, params['lidar'], 32, 512)
    img_all = range_projection(np.concatenate([pc_, pc_kf1_local, pc_kf2_local, pc_kf4_local], axis=0), params['lidar'], 32, 512)
    img_all2 = np.zeros((32, 512))
    for i in range(32):
        for j in range(512):
            img_all2[i, j] = np.max([img_[i, j], img_kf0[i, j], img_kf1[i, j], img_kf2[i, j], img_kf3[i, j], img_kf4[i, j]])

    fig, [ax1, ax2] = plt.subplots(2, 1)
    ax1.scatter(pc_[:, 0], pc_[:, 1], c='b')
    ax1.scatter(pc_kf0[:, 0], pc_kf0[:, 1], c='orange', alpha=0.5)
    ax1.scatter(pc_kf1[:, 0], pc_kf1[:, 1], c='g', alpha=0.5)
    ax1.scatter(pc_kf2[:, 0], pc_kf2[:, 1], c='m', alpha=0.5)
    ax1.scatter(pc_kf3[:, 0], pc_kf3[:, 1], c='r', alpha=0.5)
    ax1.scatter(pc_kf4[:, 0], pc_kf4[:, 1], c='y', alpha=0.5)

    ax2.scatter(pc_[:, 0], pc_[:, 1], c='b')
    ax2.scatter(pc_kf0_local[:, 0], pc_kf0_local[:, 1], c='orange', alpha=0.5)
    ax2.scatter(pc_kf1_local[:, 0], pc_kf1_local[:, 1], c='g', alpha=0.5)
    ax2.scatter(pc_kf2_local[:, 0], pc_kf2_local[:, 1], c='m', alpha=0.5)
    ax2.scatter(pc_kf3_local[:, 0], pc_kf3_local[:, 1], c='r', alpha=0.5)
    ax2.scatter(pc_kf4_local[:, 0], pc_kf4_local[:, 1], c='y', alpha=0.5)
    plt.show()

    fig, [ax1, ax2, ax3, ax4, ax5, ax6, ax7] = plt.subplots(7, 1)
    ax1.imshow(img_)
    ax2.imshow(img_kf0)
    ax3.imshow(img_kf1)
    ax4.imshow(img_kf2)
    ax5.imshow(img_kf3)
    ax6.imshow(img_kf4)
    ax7.imshow(img_all)
    plt.show()

    img_kf4_orig = range_projection(pc_kf4, params['lidar'], 32, 512)
    num_points_kf4 = len(pc_kf4_local)
    num_pixels_kf4 = np.sum(img_kf4 > -1)
    num_pixels_kf4_orig = np.sum(img_kf4_orig > -1)
    num_pixels_all = np.sum(img_all > -1)

    print(f'points in keyframe 4: {num_points_kf4}')
    print(f'pixels in keyframe 4: {num_pixels_kf4}')
    print(f'pixels in keyframe 4 original: {num_pixels_kf4_orig}')
    print(f'pixels in all: {num_pixels_all}')
