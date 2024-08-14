import os
import torch
import numpy as np
import open3d as o3d
import rospy

from nn.compute_descriptor import compute_descriptors
from nn.projection_light import RangeProjection
from nn.overlap_transformer import OverlapTransformer32


class PointCloudDescriptor:
    def __init__(self):
        self.projector = RangeProjection()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = OverlapTransformer32().to(self.device)

    def getDescriptor(self, pcd_path):
        pc = np.asarray(o3d.io.read_point_cloud(pcd_path).points)
        img = self.projector.doProjection(pc)
        descriptor = compute_descriptors(self.model, img)

        return descriptor

def callback(msg, model):
    img = torch.tensor(msg).unsqueeze(0).unsqueeze(0).float().to('cuda')
    descriptor = model(img)

    print(descriptor.shape)


if __name__ == '__main__':
    pcd_path = '/media/vectr/vectr3/Dataset/overlap_transformer/pcd_files/botanical_garden/1654559824.672178760.pcd'
    pc = np.asarray(o3d.io.read_point_cloud(pcd_path).points)
    print(pc.shape)

    model = OverlapTransformer32().to('cuda')
    projector = RangeProjection()

    img = projector.doProjection(pc)
    print(img.shape)
    callback(img, model)
