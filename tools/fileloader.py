import os
import numpy as np
import torch
import cv2
from matplotlib import image as mpimg


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


def load_images(folder_path):
    full_path = os.path.expanduser(folder_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f'File {full_path} does not exist!')


def load_poses(file_path):
    full_path = os.path.expanduser(file_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f'File {full_path} does not exist!')


def load_descriptors(file_path):
    full_path = os.path.expanduser(file_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f'File {full_path} does not exist!')


def load_labels(file_path):
    full_path = os.path.expanduser(file_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f'File {full_path} does not exist!')


def read_image(image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    depth_data = torch.from_numpy(img).type(torch.float32).to(device)
    depth_data = torch.unsqueeze(depth_data, dim=0)
    depth_data = torch.unsqueeze(depth_data, dim=0)
    return depth_data


if __name__ == '__main__':
    file_path = '/media/vectr/T9/Dataset/overlap_transformer/npy_files/bomb_shelter'
    img_path = '/media/vectr/T9/Dataset/overlap_transformer/png_files/bomb_shelter/depth/000000.png'
    file_paths = load_files(file_path)
    img = read_image(img_path)
