"""
File to generate the range image.

The original file is developed by Xieyuanli Chen and Thomas Läbe. More detail is available
https://github.com/PRBonn/OverlapNet
"""
import os
import sys
import numpy as np
import open3d as o3d
import cv2
import tqdm
import yaml
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

from tools.fileloader import load_files, read_pc


def range_projection(points, params, proj_H=32, proj_W=900):
    """
    Project a point cloud into a spherical projection, range image.

    Args:
        points: (numpy.array) point cloud.
        params: (dict) parameters of the lidar.
        proj_H: (int) projection height.
        proj_W: (int) projection width.
    Returns:
        proj_range: projected range image with depth, each pixel contains the closest corresponding depth.
    """
    # lidar view parameters
    fov_up = params['fov_up'] / 180.0 * np.pi  # field of view up in radians
    fov_down = params['fov_down'] / 180.0 * np.pi  # field of view down in radians
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians

    # get depth of all points
    depth = np.linalg.norm(points[:, :3], axis=1)
    points = points[(depth > params['eps']) & (depth < params['max_range_0.8'])]
    depth = depth[(depth > params['eps']) & (depth < params['max_range_0.8'])]

    # get scan components
    points_x = points[:, 0]
    points_y = points[:, 1]
    points_z = points[:, 2]

    # get angles of all points
    yaw = -np.arctan2(points_y, points_x)
    pitch = np.arcsin(points_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

    # order in decreasing depth
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    proj_range = np.full((proj_H, proj_W), -1, dtype=np.float32)  # [H,W] range (-1 is no data)
    proj_range[proj_y, proj_x] = depth
    return proj_range


def gen_range_images(src_folder_path, dst_folder_path, params, proj_H, proj_W, normalize=False):
    """
        Generate projected range data in the shape of (proj_H, proj_W, 1).
        The input raw data are in the shape of (Num_points, 3).
        The default LiDAR parameters are from Ouster OS1 (rev6)
    """
    # specify the goal folder
    if not os.path.exists(dst_folder_path):
        print(f'Creating new folder: {dst_folder_path}.')
        os.makedirs(dst_folder_path)
    print(f'Generating range images in {dst_folder_path}.')

    # load LiDAR scan files
    pc_paths = load_files(src_folder_path)

    # iterate over all scan files
    for i in tqdm.tqdm(range(len(pc_paths))):
        points = read_pc(pc_paths[i], format='numpy')
        # points = np.load(pc_paths[i]).astype(np.float32)  # for keyframe

        # generate range image
        proj_range = range_projection(points, params, proj_H, proj_W)

        # normalize the image
        if normalize:
            proj_range = proj_range / np.max(proj_range)

        # save the projection as an image
        filename = os.path.join(dst_folder_path, f'{str(i).zfill(6)}.png')
        cv2.imwrite(filename, proj_range)


if __name__ == '__main__':
    # load configuration and parameters
    config_path = '/home/vectr/PycharmProjects/lidar_learning/configs/config.yml'
    parameters_path = '/home/vectr/PycharmProjects/lidar_learning/configs/parameters.yml'

    config = yaml.safe_load(open(config_path))
    parameters = yaml.safe_load(open(parameters_path))

    # choose sequence
    seqs = config['seqs']['all']
    seq = seqs[9] # os0(45): 0, 5, 6, 7; os1(22.5): 1, 2, 3, 4, 8, 9

    # load point clouds path
    # pcd_files_path = os.path.join(config['data_root']['keyframes'], seq, 'npy_files')
    # pcd_files_path = os.path.join(config['data_root']['keyframes'], seq, 'pcd_files')
    pcd_files_path = os.path.join(config['data_root']['pcd_files'], seq)

    # load the destination path
    png_files_path = os.path.join(config['data_root']['keyframes'], seq, 'png_files', '1024')
    # png_files_path = os.path.join(config['data_root']['png_files'], '1024', seq)

    # lidar parameters
    lidar_params = parameters['lidar']

    # projection parameters
    proj_H = 32
    proj_W = 1024

    # generate range images
    gen_range_images(pcd_files_path, png_files_path, lidar_params, proj_H, proj_W)

    # # view generated image
    # img = mpimg.imread(os.path.join(png_files_path, '001800.png'))
    # plt.imshow(img, cmap='viridis')
    # plt.show()
