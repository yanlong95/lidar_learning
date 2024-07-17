"""
Original code from: https://github.com/valeoai/rangevit/blob/main/dataset/range_view_loader.py with modifications.
"""
import os
import numpy as np
import yaml

from tools.fileloader import read_pc, load_files


class SemanticKitt:
    def __init__(self, dataset_path, sequences):
        self.dataset_path = dataset_path
        self.sequences = sequences

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")

        self.sequences.sort()

        self.pointcloud_files = []
        for seq in self.sequences:
            files_path = os.path.join(self.dataset_path, f'{str(seq).zfill(2)}', 'velodyne')
            files = load_files(files_path)
            self.pointcloud_files.extend(files)

        print('Total number of pointcloud files:', len(self.pointcloud_files))

    @staticmethod
    def readPCD(path):
        return read_pc(path)

    def loadDataByIndex(self, idx):
        pointcloud = self.readPCD(self.pointcloud_files[idx])
        return pointcloud

    # for debugging
    def __getitem__(self, idx):
        return self.pointcloud_files[idx]

    def __len__(self):
        return len(self.pointcloud_files)


class OverlapKitti:
    def __init__(self, dataset_path, sequences):
        self.dataset_path = dataset_path
        self.sequences = sequences

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")

        self.sequences.sort()

        self.pointcloud_files = []
        for seq in self.sequences:
            files_path = os.path.join(self.dataset_path, f'{str(seq).zfill(2)}', 'velodyne')
            files = load_files(files_path)
            self.pointcloud_files.extend(files)

        print('Total number of pointcloud files:', len(self.pointcloud_files))

    @staticmethod
    def readPCD(path):
        return read_pc(path)

    def loadDataByIndex(self, idx):
        pointcloud = self.readPCD(self.pointcloud_files[idx])
        return pointcloud

    # for debugging
    def __getitem__(self, idx):
        return self.pointcloud_files[idx]

    def __len__(self):
        return len(self.pointcloud_files)


if __name__ == '__main__':
    dataset_path = '/media/vectr/T7/Datasets/public_datasets/kitti/dataset/sequences'
    sequences = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    config_path = '/home/vectr/PycharmProjects/lidar_learning/data/kitti/dataset/config_kitti.yml'

    kitti = SemanticKitt(dataset_path, sequences, config_path)
