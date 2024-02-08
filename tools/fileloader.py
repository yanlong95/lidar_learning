import os
import numpy as np
import torch
import cv2
from matplotlib import pyplot as plt


def load_files(folder_path):
    """
    Load all files in a folder.

    Args:
        folder_path: (string) the path of the folder containing the files
    Returns:
        files_paths: (list) the paths of the files
    """
    # check if the directory exist
    full_path = os.path.expanduser(folder_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f'File {full_path} does not exist!')

    # load all files
    file_paths = [os.path.join(root, file) for root, dirs, files in os.walk(full_path) for file in files]
    file_paths.sort()
    return file_paths


def load_poses(file_path):
    """
    Load the positions file (.txt or .npz).

    Args:
        file_path: (string) the path of the file containing the positions
    Returns:
        positions: (np.array) the positions in shape (n, 3, 4).
    """
    full_path = os.path.expanduser(file_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f'File {full_path} does not exist!')

    _, ext = os.path.splitext(full_path)
    if ext == '.txt':
        poses = np.loadtxt(file_path, delimiter=' ', dtype=np.float32).reshape(-1, 3, 4)
    elif ext == '.npz':
        poses = np.load(full_path, allow_pickle=True)['arr_0'].astype(np.float32)
        # poses = np.load(full_path, allow_pickle=True)['poses'].astype(np.float32)
    else:
        raise TypeError(f'File {full_path} cannot read!')
    return poses


def load_descriptors(file_path):
    full_path = os.path.expanduser(file_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f'File {full_path} does not exist!')


def load_labels(file_path):
    full_path = os.path.expanduser(file_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f'File {full_path} does not exist!')


def read_image(image_path):
    """
    Read a single image in tensor form.

    Args:
        image_path: (string) the path of the image
    Returns:
        depth_data: (tensor) image tensor in shape [1, 1, H, W]
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)                    # in grayscale, shape (H, W)
    depth_data = torch.from_numpy(image).type(torch.float32).to(device)
    depth_data = torch.unsqueeze(depth_data, dim=0)                         # shape (1, H, W)
    depth_data = torch.unsqueeze(depth_data, dim=0)                         # shape (1, 1, H, W)
    return depth_data


if __name__ == '__main__':
    file_path = '/media/vectr/T9/Dataset/overlap_transformer/npy_files/bomb_shelter'
    img_path = '/media/vectr/T9/Dataset/overlap_transformer/png_files/bomb_shelter/depth/000000.png'
    poses_path = '/media/vectr/T9/Dataset/overlap_transformer/poses/bomb_shelter/poses.txt'
    load_poses(poses_path)

