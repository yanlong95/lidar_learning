import os
import yaml
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from tools.fileloader import load_files, read_pc, load_poses, load_xyz_rot, load_overlaps
from data.kitti.dataset import projection
from data.compute_matching_points import PointMatcher


if __name__ == '__main__':
    config_path = '/home/vectr/PycharmProjects/lidar_learning/configs/config.yml'
    params_path = '/home/vectr/PycharmProjects/lidar_learning/configs/parameters.yml'

    config = yaml.safe_load(open(config_path))
    params = yaml.safe_load(open(params_path))

    pcd_paths = load_files('/media/vectr/vectr3/Dataset/overlap_transformer/pcd_files/botanical_garden')
    poses = load_poses('/media/vectr/vectr3/Dataset/overlap_transformer/poses/botanical_garden/poses.txt')
    xyz, _ = load_xyz_rot('/media/vectr/vectr3/Dataset/overlap_transformer/poses/botanical_garden/poses.txt')
    overlaps = load_overlaps('/media/vectr/vectr3/Dataset/overlap_transformer/overlaps/botanical_garden.bin')

    idx1 = 450
    idx2 = 510

    pc1 = read_pc(pcd_paths[idx1])
    pc2 = read_pc(pcd_paths[idx2])

    matcher = PointMatcher()
    pc1_o3d = o3d.geometry.PointCloud()
    pc1_o3d.points = o3d.utility.Vector3dVector(pc1[:, :3])
    pc1_o3d = matcher.align_points(pc1_o3d, poses[idx1])
    pc1_o3d.paint_uniform_color([1, 0, 0])
    pc2_o3d = o3d.geometry.PointCloud()
    pc2_o3d.points = o3d.utility.Vector3dVector(pc2[:, :3])
    pc2_o3d = matcher.align_points(pc2_o3d, poses[idx2])
    pc2_o3d.paint_uniform_color([0, 1, 0])
    o3d.visualization.draw_geometries([pc1_o3d, pc2_o3d])

    eval = o3d.pipelines.registration.evaluate_registration(pc1_o3d, pc2_o3d, 0.5, np.eye(4))
    print(eval)



    matcher = PointMatcher()
    overlap, _, _, _ = matcher.do_matching(pc1, pc2, poses[idx1], poses[idx2], num_neighbors=1)
    pc1_indices, pc2_indices = matcher.do_matching_indices(pc1, pc2, poses[idx1], poses[idx2])
    matcher.matching_chamfer(pc1, pc2, poses[idx1], poses[idx2])
    print('------------------')
    print(overlap)
    print(len(pc1_indices))
    print(len(np.unique(pc1_indices)))
    print(len(np.unique(pc2_indices)))

    projector = projection.RangeProjection(fov_up=22.5, fov_down=-22.5, proj_h=32, proj_w=512)
    proj_pc1, proj_img1, proj_idx1, proj_mask1 = projector.doProjection(pc1)
    proj_pc2, proj_img2, proj_idx2, proj_mask2 = projector.doProjection(pc2)

    mask1 = np.isin(proj_idx1, pc1_indices)
    mask2 = np.isin(proj_idx2, pc2_indices)
    mask3 = mask1 * mask2

    img1_masked = proj_img1 * mask3
    img2_masked = proj_img2 * mask3

    print(overlap)
    print('-----------------')
    print(len(pc1_indices))
    print(np.sum(proj_img1 > 0))
    print('-----------------')
    print(len(pc2))
    print(len(pc2_indices))
    print(np.sum(proj_img2 > 0))
    print('-----------------')
    print(np.sum(mask1))
    print(np.sum(mask2))
    print(np.sum(mask3))

    fig, [ax1, ax2, ax3, ax4] = plt.subplots(4, 1)
    ax1.imshow(proj_img1)

    ax2.imshow(proj_img2)
    ax3.imshow(img1_masked)
    ax4.imshow(mask2)
    plt.show()
