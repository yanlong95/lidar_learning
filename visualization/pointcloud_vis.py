import os
import numpy as np
import torch
import open3d as o3d
import matplotlib.pyplot as plt

from tools.fileloader import load_files, read_pc, load_poses
from point_based.tools.utils_pointcloud import transform_point_cloud
from tools.projection import RangeProjection
from modules.local_encoders.overlap_net_leg_32 import OverlapNetLeg32
from modules.overlap_transformer import OverlapTransformer32


def pc_vis(pc, c=None):
    """
    Visualize a point cloud.

    Args:
        pc: (numpy.ndarray) point cloud.
        c: ('string' or numpy.ndarray) color of the point cloud.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
    if c is None:
        return pcd

    if isinstance(c, np.ndarray):
        pcd.colors = o3d.utility.Vector3dVector(c)
        return pcd

    if c == 'grey':
        color = [0.5, 0.5, 0.5]
        pcd.paint_uniform_color(color)
    elif c == 'red' or c == 'r':
        color = [1, 0, 0]
        pcd.paint_uniform_color(color)
    elif c == 'green' or c == 'g':
        color = [0, 1, 0]
        pcd.paint_uniform_color(color)
    elif c == 'blue' or c == 'b':
        color = [0, 0, 1]
        pcd.paint_uniform_color(color)
    elif c == 'yellow' or c == 'y':
        color = [1, 1, 0]
        pcd.paint_uniform_color(color)
    elif c == 'distance':
        dists = np.linalg.norm(pc[:, :3], axis=1)
        dists_norm = (dists - np.min(dists)) / (np.max(dists) - np.min(dists))
        color_map = plt.get_cmap('rainbow')
        colors = color_map(dists_norm)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        raise ValueError(f'Color {c} not supported.')

    return pcd


def range_image_vis(img, show=True, saving_path=None):
    """
    Visualize a range image.

    Args:
        img: (numpy.array) range image.
        show: (bool) whether to show the image.
        saving_path: (string) path to save the image.
    """
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    if saving_path:
        plt.savefig(saving_path)
    if show:
        plt.show()


if __name__ == '__main__':
    # pcd_folder = '/Volumes/vectr7/arl/pcd_files/parking-lot'
    # poses_path = '/Volumes/vectr7/arl/poses/parking-lot/poses.txt'
    # pcd_files = load_files(pcd_folder)
    # poses = load_poses(poses_path)
    #
    # idx1 = 200
    # idx2 = 500
    #
    # pc1 = read_pc(pcd_files[idx1])
    # pc2 = read_pc(pcd_files[idx2])
    #
    # pose1 = poses[idx1]
    # pose2 = poses[idx2]
    #
    # projector = RangeProjection(proj_h=32, proj_w=512, fov_up=22.5, fov_down=-22.5, max_range=-1)
    # _, img1, _, _ = projector.doProjection(pc1)
    # _, img2, _, _ = projector.doProjection(pc2)
    #
    # # view range image
    # # range_image_vis(img1, show=True)
    # # range_image_vis(img2, show=True)
    #
    # # in global coordinate
    # pc1 = transform_point_cloud(pc1, pose1)
    # pc2 = transform_point_cloud(pc2, pose2)
    #
    # # view point cloud
    # pc1_o3d = pc_vis(pc1, 'grey')
    # pc2_o3d = pc_vis(pc2, 'green')
    # # o3d.visualization.draw_geometries([pc1_o3d, pc2_o3d])
    #
    # # local descriptors
    # model = OverlapNetLeg32().to('cpu')
    # model.eval()
    # img1_tensor = torch.from_numpy(img1).type(torch.float32).unsqueeze(0).unsqueeze(0).to('cpu')
    # with torch.no_grad():
    #     local_descriptors1 = model(img1_tensor).to('cpu').detach().numpy()
    #     local_descriptors1_i = local_descriptors1[0, 0, :, :].transpose()
    #     local_descriptors1_i = np.tile(local_descriptors1_i, (32, 1))
    #     plt.imshow(local_descriptors1_i)
    #     plt.show()

    pcd_folder = '/Volumes/vectr7/newer_college/long_experiment/pcd_files'
    pcd_files = load_files(pcd_folder)

    pc = read_pc(pcd_files[900])
    pc1_o3d = pc_vis(pc, 'distance')
    o3d.visualization.draw_geometries([pc1_o3d])

    projector = RangeProjection(proj_h=32, proj_w=512, fov_up=16.6, fov_down=-16.6, max_range=-1)
    _, img1, _, _ = projector.doProjection(pc)
    range_image_vis(img1, show=True)

    # local descriptors
    model = OverlapNetLeg32().to('cpu')
    model.eval()
    img1_tensor = torch.from_numpy(img1).type(torch.float32).unsqueeze(0).unsqueeze(0).to('cpu')
    with torch.no_grad():
        local_descriptors1 = model(img1_tensor).to('cpu').detach().numpy()
        local_descriptors1_i = local_descriptors1[0, 0, :, :].transpose()
        local_descriptors1_i = np.tile(local_descriptors1_i, (32, 1))
        plt.imshow(local_descriptors1_i)
        plt.show()
