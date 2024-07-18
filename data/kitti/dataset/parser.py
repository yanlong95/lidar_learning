"""
Original code from: https://github.com/valeoai/rangevit/blob/main/dataset/range_view_loader.py with modifications.
"""
import os
import numpy as np
import yaml

from tools.fileloader import read_pc, load_files, load_overlaps


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
        self.overlaps_batches = []
        for seq in self.sequences:
            files_path = os.path.join(self.dataset_path, f'{str(seq).zfill(2)}', 'velodyne')
            batch_path = os.path.join(self.dataset_path, f'{str(seq).zfill(2)}', 'overlaps/overlaps_batch.npz')
            files = load_files(files_path)
            overlaps_batch = load_overlaps(batch_path)
            self.pointcloud_files.extend(files)
            self.overlaps_batches.extend(overlaps_batch)

        self.overlaps_batches = np.asarray(self.overlaps_batches)
        print('Total number of pointcloud files:', len(self.pointcloud_files))

    @staticmethod
    def readPCD(path):
        return read_pc(path)

    def loadDataByIndex(self, idx, max_num_pos, max_num_neg):
        # anchor pointcloud
        anchor_pointcloud = self.readPCD(self.pointcloud_files[idx])

        # positive and negative point clouds
        batch = self.overlaps_batches[idx]
        num_pos = batch[3]
        num_neg = batch[4]

        # get indices in order of all point clouds (indices in self.pointcloud_files)
        global_index = idx - batch[0]
        pos_local_indices = batch[5:5+num_pos]
        neg_local_indices = batch[105:105+num_neg:]

        pos_indices = pos_local_indices + global_index
        neg_indices = neg_local_indices + global_index

        # random select positive and negative point clouds
        num_pos = min(num_pos, max_num_pos)
        num_neg = min(num_neg, max_num_neg)
        pos_indices = np.random.choice(pos_indices, num_pos, replace=False)
        neg_indices = np.random.choice(neg_indices, num_neg, replace=False)

        pos_pointclouds = [self.readPCD(self.pointcloud_files[i]) for i in pos_indices]
        neg_pointclouds = [self.readPCD(self.pointcloud_files[i]) for i in neg_indices]

        return anchor_pointcloud, pos_pointclouds, neg_pointclouds

    # for debugging
    def __getitem__(self, idx):
        return self.pointcloud_files[idx]

    def __len__(self):
        return len(self.pointcloud_files)


if __name__ == '__main__':
    dataset_path = '/media/vectr/T7/Datasets/public_datasets/kitti/dataset/sequences'
    sequences = [0, 1, 2]
    config_path = '/home/vectr/PycharmProjects/lidar_learning/data/kitti/dataset/config_kitti.yml'

    kitti = OverlapKitti(dataset_path, sequences)
    kitti.loadDataByIndex(0, 30, 30)
