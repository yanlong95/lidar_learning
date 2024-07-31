import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

from tools.fileloader import load_files, read_pc
from data.kitti.dataset import projection


if __name__ == '__main__':
    # load point cloud data
    data_path = '/Volumes/T7/Datasets/public_datasets/kitti/dataset/sequences/00/velodyne'
    files = load_files(data_path)
    pc = read_pc(files[0])

    projector = projection.RangeProjection()
    proj_pc, proj_img, proj_idx, proj_mask = projector.doProjection(pc)

    points = proj_pc[1, :5, :]
    dists = np.linalg.norm(points[:, :3], 2, axis=1)
