"""
Original code from: https://github.com/valeoai/rangevit/blob/main/dataset/range_view_loader.py with modifications.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from dataclasses import dataclass, field, fields

from augmentor import Augmentor, AugmentParams
from projection import RangeProjection, ScanProjection


class RangeViewLoader(Dataset):
    def __init__(self, dataset, config, data_len=-1, is_train=True, depth_only=True, normalize=False, pc1=None, pc2=None):
        self.dataset = dataset
        self.config = config
        self.is_train = is_train
        self.data_len = data_len
        self.depth_only = depth_only
        self.normalize = normalize

        self.pointcloud1 = pc1
        self.pointcloud2 = pc2

        augment_params = AugmentParams()
        augment_pc_config = self.config['augmentation_pointcloud']
        augment_img_config = self.config['augmentation_image']
        projection_config = self.config['sensor']

        # Point cloud augmentations and image augmentations parameters assignment
        if self.is_train:
            augment_params = AugmentParams()
            field_name = {f.name for f in fields(AugmentParams) if f.init}
            for key, value in augment_pc_config.items():
                if key in field_name:
                    setattr(augment_params, key, value)

            for key, value in augment_img_config.items():
                if key in field_name:
                    setattr(augment_params, key, value)

            self.augmentor = Augmentor(augment_params, augment_pc_config['do_pc_aug'], augment_img_config['do_img_aug'],
                                       augment_img_config['do_img_crop'])
        else:
            self.augmentor = None

        # Projection
        self.scan_proj = projection_config.get('scan_proj', False)
        if self.scan_proj:
            print('Use scan-based range projection.')
            self.projection = ScanProjection(
                proj_h=projection_config['proj_h'], proj_w=projection_config['proj_w'],
            )
        else:
            self.projection = RangeProjection(
                fov_up=projection_config['fov_up'], fov_down=projection_config['fov_down'],
                fov_left=projection_config['fov_left'], fov_right=projection_config['fov_right'],
                proj_h=projection_config['proj_h'], proj_w=projection_config['proj_w'],
                max_range=projection_config['max_range']
            )

        # Image normalization
        self.proj_img_mean = torch.tensor(self.config['sensor']['img_mean'], dtype=torch.float)
        self.proj_img_stds = torch.tensor(self.config['sensor']['img_stds'], dtype=torch.float)

        # Image augmentations
        if self.is_train:
            self.crop_size = augment_img_config['image_size']
            self.aug_ops = T.Compose([
                T.RandomCrop(
                    size=(augment_img_config['image_size'][0],
                          augment_img_config['image_size'][1])),
            ])
        else:
            self.crop_size = augment_img_config['original_image_size']
            self.aug_ops = T.Compose([
                T.CenterCrop((augment_img_config['original_image_size'][0],
                              augment_img_config['original_image_size'][1]))
            ])

    def __getitem__(self, index):
        '''
        proj_feature_tensor: CxHxW
        proj_sem_label_tensor: HxW
        proj_mask_tensor: HxW
        '''
        # sample data
        # pointcloud, _, _ = self.dataset.loadDataByIndex(index)
        pointcloud = self.pointcloud1

        # do point cloud augmentation
        if self.is_train and self.augmentor.do_pc_aug:
            pointcloud = self.augmentor.doAugmentationPointcloud(pointcloud)

        # do projection
        if self.depth_only:
            _, proj_range, _, _ = self.projection.doProjection(pointcloud)
            proj = proj_range[np.newaxis, ...]
        else:
            proj_pointcloud, proj_range, proj_idx, proj_mask = self.projection.doProjection(pointcloud)
            proj = np.concatenate([proj_range[np.newaxis, ...], proj_pointcloud.transpose((2, 0, 1))], axis=0)

        # do image augmentation
        if self.is_train and self.augmentor.do_img_aug:
            # pointcloud_ref, _, _ = self.dataset.loadDataByIndex(np.random.randint(len(self.dataset)) - 1)
            pointcloud_ref = self.pointcloud2
            if self.depth_only:
                _, proj_range_ref, _, _ = self.projection.doProjection(pointcloud_ref)
                proj_ref = proj_range_ref[np.newaxis, ...]
            else:
                proj_pointcloud_ref, proj_range_ref, proj_idx_ref, proj_mask_ref = self.projection.doProjection(pointcloud_ref)
                proj_ref = np.concatenate([proj_range_ref[np.newaxis, ...], proj_pointcloud_ref.transpose((2, 0, 1))], axis=0)
            proj = self.augmentor.doAugmentationImage(proj, proj_ref)

        proj_tensor = torch.from_numpy(proj)
        if self.normalize:
            if self.depth_only:
                proj_tensor = (proj_tensor - self.proj_img_mean[0]) / self.proj_img_stds[0]
            else:
                proj_tensor = (proj_tensor - self.proj_img_mean) / self.proj_img_stds

        return proj_tensor

    def __len__(self):
        if self.data_len > 0 and self.data_len < len(self.dataset):
            return self.data_len
        else:
            return len(self.dataset)
