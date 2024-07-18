import numpy as np
import torch

from tools.fileloader import read_pc, load_files
from projection import RangeProjection
from range_view_loader import RangeViewLoader
from range_view_loader_batch import RangeViewLoader as RangeViewLoaderBatch
import yaml
import matplotlib.pyplot as plt
from parser import SemanticKitt, OverlapKitti

# pc1_path = '/Volumes/T7/Datasets/public_datasets/kitti/dataset/sequences/00/velodyne/001300.bin'
# pc2_path = '/Volumes/T7/Datasets/public_datasets/kitti/dataset/sequences/00/velodyne/001500.bin'
# config_path = '/Users/yanlong/PycharmProjects/lidar_learning/data/kitti/dataset/config_kitti.yml'
# pc1 = read_pc(pc1_path)
# pc2 = read_pc(pc2_path)
# config = yaml.safe_load(open(config_path))
#
# projector = RangeProjection()
# range_view_loader = RangeViewLoader(dataset=pc1_path, config=config, is_train=True, depth_only=True, normalize=True, pc1=pc1, pc2=pc2)
# print(range_view_loader.augmentor)
#
# img1 = range_view_loader[0].to('cpu').numpy()
# img1 = img1[0, :, :]
#
# _, img_ref, _, _ = projector.doProjection(pc1)
# _, img_ref2, _, _ = projector.doProjection(pc2)
#
# fig, axs = plt.subplots(3, 1)
# axs[0].imshow(img1)
# axs[1].imshow(img_ref)
# axs[2].imshow(img_ref2)
# plt.show()

dataset_path = '/media/vectr/T7/Datasets/public_datasets/kitti/dataset/sequences'
sequences = [0, 1, 2, 3, 4, 5, 6, 7, 8]
config_path = '/home/vectr/PycharmProjects/lidar_learning/data/kitti/dataset/config_kitti.yml'

# kitti = SemanticKitt(dataset_path, sequences)
# config = yaml.safe_load(open(config_path))
#
# dataloader = RangeViewLoader(dataset=kitti, config=config, is_train=True, depth_only=False, normalize=True)
# projector = RangeProjection()
#
# img1 = dataloader[0].to('cpu').numpy()
# img1 = img1[0, :, :]
#
# _, img_ref, _, _ = projector.doProjection(read_pc(kitti[0]))
# _, img_ref2, _, _ = projector.doProjection(read_pc(kitti[1000]))
#
# fig, axs = plt.subplots(3, 1)
# axs[0].imshow(img1)
# axs[1].imshow(img_ref)
# axs[2].imshow(img_ref2)
# plt.show()


kitti = OverlapKitti(dataset_path, sequences)
config = yaml.safe_load(open(config_path))

dataloader = RangeViewLoaderBatch(dataset=kitti, config=config, is_train=True, depth_only=True, normalize=False)
projector = RangeProjection()

import time

t0 = time.time()
for i in range(100):
    anchor1, pos1, neg1 = dataloader[i]
t1 = time.time()
print(t1 - t0)
print(anchor1.shape)
print(pos1.shape)
print(neg1.shape)

# a1 = anchor1[0, 0, :, :].to('cpu').numpy()
# p1 = pos1[0, 0, :, :].to('cpu').numpy()
# n1 = neg1[0, 0, :, :].to('cpu').numpy()
#
# _, img_ref, _, _ = projector.doProjection(read_pc(kitti[0]))
# _, img_ref2, _, _ = projector.doProjection(read_pc(kitti[1000]))
# _, img_ref_3, _, _ = projector.doProjection(read_pc(kitti[1673]))
#
# fig, axs = plt.subplots(3, 1)
# axs[0].imshow(n1)
# axs[1].imshow(img_ref_3)
# axs[2].imshow(img_ref)
# plt.show()
