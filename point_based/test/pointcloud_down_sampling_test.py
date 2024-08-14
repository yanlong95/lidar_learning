import os
import open3d as o3d
import numpy as np
import time

from point_based.tools.utils_func import farthest_point_sampling
from point_based.tools.fileloader import load_xyz_rot, load_files, read_pc, load_poses
from point_based.tools.compute_matching_points import PointMatcher


if __name__ == '__main__':
    pcd_folder_path = '/media/vectr/vectr3/Dataset/overlap_transformer/pcd_files/botanical_garden'
    poses_file = '/media/vectr/vectr3/Dataset/overlap_transformer/poses/botanical_garden/poses.txt'
    pcd_files = load_files(pcd_folder_path)
    poses = load_poses(poses_file)

    matcher1 = PointMatcher()
    matcher2 = PointMatcher(down_sample=True, voxel_size=0.5)

    idx1 = 450
    idx2 = 600
    pc1 = read_pc(pcd_files[idx1])
    pc2 = read_pc(pcd_files[idx2])
    pose1 = poses[idx1]
    pose2 = poses[idx2]

    overlap1, indices1, dists1, mask1 = matcher1.do_matching(pc1, pc2, pose1, pose2, num_neighbors=1)
    overlap2, indices2, dists2, mask2 = matcher2.do_matching(pc1, pc2, pose1, pose2, num_neighbors=1, verbose=True)
    print(f'Overlap before down sampling: {overlap1}')
    print(f'Overlap after voxelization: {overlap2}')
    print(f'Number of matching points before down sampling: {np.sum(mask1)}')
    print(f'Number of matching points after down sampling: {np.sum(mask2)}')

    # farthest point sampling
    t0 = time.time()
    pc1 = farthest_point_sampling(pc1, 2000)
    pc2 = farthest_point_sampling(pc2, 2000)
    t1 = time.time()
    overlap3, indices3, dists3, mask3 = matcher1.do_matching(pc1, pc2, pose1, pose2, num_neighbors=1)
    print(f'Overlap after farthest point down sampling: {overlap3}')
    print(f'Avg farthest point sampling time: {(t1 - t0) / 2}')
