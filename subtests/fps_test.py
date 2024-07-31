import open3d as o3d
import numpy as np
import yaml
import os
import matplotlib.pyplot as plt

from tools.fileloader import load_files, read_pc, load_xyz_rot
from tools.utils_func import farthest_point_sampling
from data.kitti.dataset.projection import RangeProjection


def rot_pc(pc_loc, rot_loc, xyz_loc, rot_ref, xyz_ref):
    pc_world = pc_loc @ rot_loc.T + xyz_loc
    pc_ref = pc_world @ rot_ref - xyz_ref @ rot_ref
    return pc_ref


if __name__ == '__main__':
    # pcd_paths = load_files('/media/vectr/vectr3/Dataset/overlap_transformer/pcd_files/botanical_garden')
    # pc = read_pc(pcd_paths[0])
    #
    # num_samples = 2000
    # print(pc.shape)
    #
    # pc_sampled = farthest_point_sampling(pc, num_samples)
    # print(pc_sampled.shape)
    #
    # pcd_full = o3d.geometry.PointCloud()
    # pcd_full.points = o3d.utility.Vector3dVector(pc[:, :3])
    # pcd_full.paint_uniform_color([0.5, 0.5, 0.5])
    #
    # pcd_sampled = o3d.geometry.PointCloud()
    # pcd_sampled.points = o3d.utility.Vector3dVector(pc_sampled[:, :3])
    # pcd_sampled.paint_uniform_color([1, 0.706, 0])
    #
    # # o3d.visualization.draw_geometries([pcd_sampled])
    #
    # projector = RangeProjection(proj_h=32, proj_w=512, fov_up=22.5, fov_down=-22.5)
    # _, img_full, _, _ = projector.doProjection(pc)
    # _, img_sampled, _, _ = projector.doProjection(pc_sampled)
    #
    # fig, [ax1, ax2] = plt.subplots(2, 1)
    # ax1.imshow(img_full)
    # ax2.imshow(img_sampled)
    # plt.show()

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

    pc_kf0 = farthest_point_sampling(pc_kf0, 2000)
    pc_kf1 = farthest_point_sampling(pc_kf1, 2000)
    pc_kf2 = farthest_point_sampling(pc_kf2, 2000)
    pc_kf3 = farthest_point_sampling(pc_kf3, 2000)
    pc_kf4 = farthest_point_sampling(pc_kf4, 2000)

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

    pc_ = rot_pc(pc_, rot_, xyz_, rot_, xyz_)
    pc_kf0 = rot_pc(pc_kf0, rot_kf0, xyz_kf0, rot_, xyz_)
    pc_kf1 = rot_pc(pc_kf1, rot_kf1, xyz_kf1, rot_, xyz_)
    pc_kf2 = rot_pc(pc_kf2, rot_kf2, xyz_kf2, rot_, xyz_)
    pc_kf3 = rot_pc(pc_kf3, rot_kf3, xyz_kf3, rot_, xyz_)
    pc_kf4 = rot_pc(pc_kf4, rot_kf4, xyz_kf4, rot_, xyz_)

    pc_submap_sampled = np.concatenate([pc_kf1, pc_kf2, pc_kf3, pc_kf4], axis=0)

    pcd_full = o3d.geometry.PointCloud()
    pcd_full.points = o3d.utility.Vector3dVector(pc_[:, :3])
    pcd_full.paint_uniform_color([0.5, 0.5, 0.5])

    pcd_sampled = o3d.geometry.PointCloud()
    pcd_sampled.points = o3d.utility.Vector3dVector(pc_submap_sampled[:, :3])
    pcd_sampled.paint_uniform_color([1, 0.706, 0])

    o3d.visualization.draw_geometries([pcd_full, pcd_sampled])

    projector = RangeProjection(proj_h=32, proj_w=512, fov_up=22.5, fov_down=-22.5)
    _, img_full, _, _ = projector.doProjection(pc_)
    _, img_sampled, _, _ = projector.doProjection(pc_submap_sampled)

    fig, [ax1, ax2] = plt.subplots(2, 1)
    ax1.imshow(img_full)
    ax2.imshow(img_sampled)
    plt.show()
