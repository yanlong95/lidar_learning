import os
import psutil
import yaml
import open3d as o3d

from tools.fileloader import load_files, read_pc

def compute_overlaps():
    pass


def threads_count():
    return psutil.cpu_count()


def kd_tree(pc):
    pc_tree = o3d.geometry.KDTreeFlann(pc)
    return pc_tree


def search_nearest_points(pc_tree, point, num_neighbors):
    [k, idx, _] = pc_tree.search_knn_vector_3d(point, num_neighbors)   # idx in format IntVector (o3d.utility.IntVector)
    return idx


def search_radius_points(pc_tree, point, radius):
    [k, idx, _] = pc_tree.search_radius_vector_3d(point, radius)
    return idx


if __name__ == '__main__':
    config_path = '/home/vectr/PycharmProjects/lidar_learning/configs/config.yml'
    lidar_params_path = '/home/vectr/PycharmProjects/lidar_learning/configs/config.yml'

    config = yaml.safe_load(open(config_path))
    lidar_params = yaml.safe_load(open(lidar_params_path))

    seq = 'sculpture_garden'
    pcd_folder_path = os.path.join(config['data_root']['pcd_files'], seq)
    pcd_files_path = load_files(pcd_folder_path)

    scan1 = 910
    scan2 = 1660

    # in format of o3d point cloud
    pc1 = read_pc(pcd_files_path[scan1], format='o3d')
    pc2 = read_pc(pcd_files_path[scan2], format='o3d')

    pc1_tree = kd_tree(pc1)
    pc2_tree = kd_tree(pc2)

    idx = search_nearest_points(pc1_tree, pc1.points[1000], 100)
    print(type(idx))
    print(idx)
