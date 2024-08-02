import os
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

from tools.fileloader import load_files, load_poses, read_pc
from data.compute_matching_points import PointMatcher
from data.kitti.dataset import projection


if __name__ == "__main__":
    # Load point clouds
    pc_path = "/Volumes/T7/Datasets/public_datasets/kitti/dataset/sequences/00/velodyne"
    pc_files = load_files(pc_path)

    # Load poses
    poses_path = "/Volumes/T7/Datasets/public_datasets/kitti/dataset/sequences/00/poses/poses.txt"
    poses = load_poses(poses_path)

    # Load pc
    idx1 = 100
    idx2 = 110
    pointcloud1 = read_pc(pc_files[idx1])
    pointcloud2 = read_pc(pc_files[idx2])
    pose1 = np.vstack((poses[idx1], np.array([0, 0, 0, 1])))
    pose2 = np.vstack((poses[idx2], np.array([0, 0, 0, 1])))

    # Visualize point cloud
    pc1 = o3d.geometry.PointCloud()
    pc1.points = o3d.utility.Vector3dVector(pointcloud1[:, :3])
    pc2 = o3d.geometry.PointCloud()
    pc2.points = o3d.utility.Vector3dVector(pointcloud2[:, :3])

    # pc1 = pc1.transform(pose1)
    # pc2 = pc2.transform(pose2)

    # o3d.visualization.draw_geometries([pc1, pc2])

    matcher = PointMatcher()
    indices1, indices2 = matcher.do_matching_indices(pc1, pc2, pose1, pose2)
    print(len(indices1), len(indices2))

    projector = projection.RangeProjection()
    proj_pc1, proj_img1, proj_idx1, proj_mask1 = projector.doProjection(pointcloud1)
    proj_pc2, proj_img2, proj_idx2, proj_mask2 = projector.doProjection(pointcloud2)

    mask1 = np.isin(proj_idx1, indices1)
    mask2 = np.isin(proj_idx2, indices2)
    mask3 = mask1 * mask2

    img1_masked = proj_img1 * mask3
    img2_masked = proj_img2 * mask3

    fig, [ax1, ax2, ax3, ax4] = plt.subplots(4, 1)
    ax1.imshow(proj_img1)

    ax2.imshow(proj_img2)
    ax3.imshow(img1_masked)
    ax4.imshow(img2_masked)
    plt.show()
