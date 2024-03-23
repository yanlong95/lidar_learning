"""
Script to test if the model is yaw-invariant.
"""
import os
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from tools.utils.utils import load_vertex, range_projection, load_poses, load_files
from modules.overlap_transformer_haomo import featureExtracter


def read_pc(pc_path):
    pc = o3d.io.read_point_cloud(pc_path)
    points = np.asarray(pc.points, dtype=np.float32)
    return points


def img2tensor(image_projection):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    depth_data_tensor = torch.from_numpy(image_projection).type(torch.FloatTensor).to(device)
    depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)
    depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)

    return depth_data_tensor


def calc_rot_matrix(angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])


def calc_descriptors(range_tensor, amodel):
    with torch.no_grad():
        curr_batch = torch.cat((range_tensor, range_tensor), dim=0)

        amodel.eval()
        descriptor = amodel(curr_batch)
        descriptor = descriptor[0, :].cpu().detach().numpy()

    return descriptor


if __name__ == '__main__':
    seq = 'sculpture_garden'
    scan1 = 910
    scan2 = 1660

    poses_path = os.path.join('/media/vectr/T9/Dataset/overlap_transformer/poses/', seq, 'poses.txt')
    pcd_folder_path = os.path.join('/media/vectr/T9/Dataset/overlap_transformer/pcd_files/', seq)
    weights_path = os.path.join('/media/vectr/T9/Dataset/overlap_transformer/weights/'
                                'pretrained_overlap_transformer_full_test50.pth.tar')

    pcd_files_path = load_files(pcd_folder_path)
    poses = load_poses(poses_path)

    # # check 2 scans positions
    # xy = poses[:, :2, 3]
    # xy_scan1 = poses[scan1, :2, 3]
    # xy_scan2 = poses[scan2, :2, 3]
    # plt.plot(xy[:, 0], xy[:, 1], c='b', label='trajectory')
    # plt.scatter(xy_scan1[0], xy_scan1[1], c='r', label='scan1')
    # plt.scatter(xy_scan2[0], xy_scan2[1], c='g', label='scan2')
    # plt.show()
    # print(f'distance between scan1 and scan2: {np.linalg.norm(xy_scan1 - xy_scan2):.3f}.')

    # load point cloud
    pc1 = read_pc(pcd_files_path[scan1])
    pc2 = read_pc(pcd_files_path[scan2])

    # rotate point cloud
    theta0 = np.pi * 0
    theta1 = np.pi / 2
    theta2 = np.pi
    theta3 = np.pi * 3 / 2
    theta4 = np.pi * 2
    rot_matrix0 = calc_rot_matrix(theta0)
    rot_matrix1 = calc_rot_matrix(theta1)
    rot_matrix2 = calc_rot_matrix(theta2)
    rot_matrix3 = calc_rot_matrix(theta3)
    rot_matrix4 = calc_rot_matrix(theta4)

    pc1_rot0 = (rot_matrix0 @ pc1.T).T
    pc1_rot1 = (rot_matrix1 @ pc1.T).T
    pc1_rot2 = (rot_matrix2 @ pc1.T).T
    pc1_rot3 = (rot_matrix3 @ pc1.T).T
    pc1_rot4 = (rot_matrix4 @ pc1.T).T

    # projection
    range_proj0 = range_projection(pc1_rot0, fov_up=45.0, fov_down=-45.0, proj_H=32, proj_W=900, max_range=35)
    range_proj1 = range_projection(pc1_rot1, fov_up=45.0, fov_down=-45.0, proj_H=32, proj_W=900, max_range=35)
    range_proj2 = range_projection(pc1_rot2, fov_up=45.0, fov_down=-45.0, proj_H=32, proj_W=900, max_range=35)
    range_proj3 = range_projection(pc1_rot3, fov_up=45.0, fov_down=-45.0, proj_H=32, proj_W=900, max_range=35)
    range_proj4 = range_projection(pc1_rot4, fov_up=45.0, fov_down=-45.0, proj_H=32, proj_W=900, max_range=35)

    # to tensor
    range_tensor0 = img2tensor(range_proj0)
    range_tensor1 = img2tensor(range_proj1)
    range_tensor2 = img2tensor(range_proj2)
    range_tensor3 = img2tensor(range_proj3)
    range_tensor4 = img2tensor(range_proj4)

    # calculate feature vectors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(weights_path, map_location=device)
    amodel = featureExtracter(height=32, width=900, channels=1, use_transformer=True).to(device)
    amodel.load_state_dict(checkpoint['state_dict'])

    feature0 = calc_descriptors(range_tensor0, amodel)
    feature1 = calc_descriptors(range_tensor1, amodel)
    feature2 = calc_descriptors(range_tensor2, amodel)
    feature3 = calc_descriptors(range_tensor3, amodel)
    feature4 = calc_descriptors(range_tensor4, amodel)

    print(f'Normal distance for 90 deg rotation: {np.linalg.norm(feature0 - feature1) / np.linalg.norm(feature0)}.')
    print(f'Normal distance for 180 deg rotation: {np.linalg.norm(feature0 - feature2) / np.linalg.norm(feature0)}.')
    print(f'Normal distance for 270 deg rotation: {np.linalg.norm(feature0 - feature3) / np.linalg.norm(feature0)}.')
    print(f'Normal distance for 360 deg rotation: {np.linalg.norm(feature0 - feature4) / np.linalg.norm(feature0)}.')

    print(np.linalg.norm(feature0))
