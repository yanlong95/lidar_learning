import os
import numpy as np
import torch
import cv2
import open3d as o3d


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
    Load the positions file (.txt, .npz or .npy).

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
    elif ext == '.npy':
        poses = np.load(full_path, allow_pickle=True).astype(np.float32)
    else:
        raise TypeError(f'Positions file {full_path} cannot be read!')
    return poses


def load_xyz_rot(file_path):
    """
        Load the xyz positions and rotations from the file (.txt or .npz).

        Args:
            file_path: (string) the path of the file containing the positions
        Returns:
            xyz: (np.array) the positions in shape (n, 3, 1).
            rot: (np.array) the rotation matrix in shape (n, 3, 3)
        """
    poses = load_poses(file_path)
    xyz = poses[:, :, -1]
    rot = poses[:, :, :3]
    return xyz, rot


def load_descriptors(file_path):
    """
    Load the descriptors file (.txt or .npy).

    Args:
        file_path: (string) the path of the file containing the descriptors
    Returns:
        descriptors: (np.array) the descriptors in shape (n, 256).
    """
    full_path = os.path.expanduser(file_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f'File {full_path} does not exist!')

    _, ext = os.path.splitext(full_path)
    if ext == '.txt':
        descriptors = np.loadtxt(file_path, delimiter=' ', dtype=np.float32)
    elif ext == '.npy':
        descriptors = np.load(full_path, allow_pickle=True).astype(np.float32)
    else:
        raise TypeError(f'Descriptors file {full_path} cannot be read!')

    return descriptors


def read_pc(pc_path, format='numpy'):
    """
    Read a single point cloud in numpy form.

    Args:
        pc_path: (string) the path of the point cloud file (pcd file).
    Returns:
        points: (numpy array) points in shape (n, 3)
    """
    pc = o3d.io.read_point_cloud(pc_path)
    if format == 'numpy':
        pc = np.asarray(pc.points, dtype=np.float32)

    return pc


def read_image(image_path, device='cuda'):
    """
    Read a single image in tensor form.

    Args:
        image_path: (string) the path of the image
        device: (string or torch.device()) training device
    Returns:
        depth_data: (tensor) image tensor in shape (1, 1, H, W)
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)                    # in grayscale, shape (H, W)
    depth_data = torch.from_numpy(image).type(torch.float32).to(device)
    depth_data = torch.unsqueeze(depth_data, dim=0)                         # shape (1, H, W)
    depth_data = torch.unsqueeze(depth_data, dim=0)                         # shape (1, 1, H, W)
    return depth_data


if __name__ == '__main__':
    file_path = '/media/vectr/T9/Dataset/overlap_transformer/npy_files/bomb_shelter'
    img_path = '/media/vectr/T9/Dataset/overlap_transformer/png_files/bomb_shelter/depth/000000.png'
    poses_path = '/media/vectr/T9/Dataset/overlap_transformer/poses/bomb_shelter/poses.txt'
    descriptors_path = '/media/vectr/T9/Dataset/overlap_transformer/descriptors/bomb_shelter/keyframe_descriptors.npy'
    pc_path = '/media/vectr/T9/Dataset/overlap_transformer/pcd_files/bomb_shelter/1698379586.665777408.pcd'
    p = read_pc(pc_path)
    print(p.shape)
