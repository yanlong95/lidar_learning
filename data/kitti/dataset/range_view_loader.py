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
    def __init__(self, dataset, config, data_len=-1, is_train=True, return_uproj=False, depth_only=True):
        self.dataset = dataset
        self.config = config
        self.is_train = is_train
        self.data_len = data_len
        self.return_uproj = return_uproj
        self.depth_only = depth_only

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
        pointcloud, _, _ = self.dataset.loadDataByIndex(index)

        if self.depth_only:
            _, proj_range, _, _ = self.projection.doProjection(pointcloud)
            proj = proj_range[np.newaxis, ...]
        else:
            proj_pointcloud, proj_range, proj_idx, proj_mask = self.projection.doProjection(pointcloud)
            proj = np.concatenate([proj_range[np.newaxis, ...],
                                   np.transpose(proj_pointcloud, (2, 0, 1)),
                                   proj_idx[np.newaxis, ...],
                                   proj_mask[np.newaxis, ...]], axis=0)

        if self.is_train and self.augmentor.do_pc_aug:
            pointcloud = self.augmentor.doAugmentationPointcloud(pointcloud)  # n, 4
        if self.depth_only:
            _, proj_range, _, _ = self.projection.doProjection(pointcloud)
        else:
            proj_pointcloud, proj_range, proj_idx, proj_mask = self.projection.doProjection(pointcloud)
            proj = np.concatenate([proj_range[np.newaxis, ...],
                                   proj_pointcloud,
                                   proj_idx[np.newaxis, ...],
                                   proj_mask[np.newaxis, ...]], axis=0)

        # get a reference point cloud and range image if do range image augmentation
        if self.is_train and self.augmentor.do_img_aug:
            pointcloud_ref, _, _ = self.dataset.loadDataByIndex(np.random.randint(len(self.dataset)) - 1)
            proj_pointcloud_ref, proj_range_ref, proj_idx_ref, proj_mask_ref = self.projection.doProjection(pointcloud_ref)
            proj_ref = np.concatenate([proj_range_ref[np.newaxis, ...],
                                       proj_pointcloud_ref,
                                       proj_idx_ref[np.newaxis, ...],
                                       proj_mask_ref[np.newaxis, ...]], axis=0)
            if self.depth_only:
                proj_range = self.augmentor.doAugmentationImage(proj_range, proj_range_ref)
            else:
                proj = self.augmentor.doAugmentationImage(proj, proj_ref)

        proj_range_tensor = torch.from_numpy(proj_range)
        proj_xyz_tensor = torch.from_numpy(proj_pointcloud[..., :3])
        proj_intensity_tensor = torch.from_numpy(proj_pointcloud[..., 3])
        proj_intensity_tensor = proj_intensity_tensor.ne(-1).float() * proj_intensity_tensor
        proj_feature_tensor = torch.cat(
            [proj_range_tensor.unsqueeze(0), proj_xyz_tensor.permute(2, 0, 1), proj_intensity_tensor.unsqueeze(0)], 0)

        proj_feature_tensor = (proj_feature_tensor - self.proj_img_mean[:, None, None]) / self.proj_img_stds[:, None,
                                                                                          None]
        proj_feature_tensor = proj_feature_tensor * proj_mask_tensor.unsqueeze(0).float()



        # Data augmentation
        if self.augmentor.do_img_crop:
            proj_tensor = self.aug_ops(proj_tensor)

        return proj_tensor[0:5], proj_tensor[5], proj_tensor[6]

    def __len__(self):
        if self.data_len > 0 and self.data_len < len(self.dataset):
            return self.data_len
        else:
            return len(self.dataset)
