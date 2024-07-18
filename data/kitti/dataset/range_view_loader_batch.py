"""
Original code from: https://github.com/valeoai/rangevit/blob/main/dataset/range_view_loader.py with modifications.
"""
import numpy as np
import einops
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from dataclasses import dataclass, field, fields

from augmentor import Augmentor, AugmentParams
from projection import RangeProjection, ScanProjection


class RangeViewLoader(Dataset):
    def __init__(self, dataset, config, data_len=-1, is_train=True, depth_only=True, normalize=False):
        self.dataset = dataset
        self.config = config
        self.is_train = is_train
        self.data_len = data_len
        self.depth_only = depth_only
        self.normalize = normalize

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
        max_num_pos = self.config['batch']['max_num_pos']
        max_num_neg = self.config['batch']['max_num_neg']
        anchor_pc, pos_pc, neg_pc = self.dataset.loadDataByIndex(index, max_num_pos, max_num_neg)

        # do point cloud augmentation
        if self.is_train and self.augmentor.do_pc_aug:
            anchor_pc = self.augmentor.doAugmentationPointcloud(anchor_pc)
            pos_pc = [self.augmentor.doAugmentationPointcloud(pc) for pc in pos_pc]
            neg_pc = [self.augmentor.doAugmentationPointcloud(pc) for pc in neg_pc]

        # do projection
        if self.depth_only:
            _, anchor_proj_range, _, _ = self.projection.doProjection(anchor_pc)
            anchor_proj = anchor_proj_range[np.newaxis, ...]
            pos_proj = []
            neg_proj = []
            for pc in pos_pc:
                _, pos_proj_range, _, _ = self.projection.doProjection(pc)
                pos_proj.append(pos_proj_range[np.newaxis, ...])
            for pc in neg_pc:
                _, neg_proj_range, _, _ = self.projection.doProjection(pc)
                neg_proj.append(neg_proj_range[np.newaxis, ...])
        else:
            anchor_proj_pointcloud, anchor_proj_range, _, _ = self.projection.doProjection(anchor_pc)
            anchor_proj = np.concatenate([anchor_proj_range[np.newaxis, ...],
                                          anchor_proj_pointcloud.transpose((2, 0, 1))], axis=0)
            pos_proj = []
            neg_proj = []
            for pc in pos_pc:
                pos_proj_pointcloud, pos_proj_range, _, _ = self.projection.doProjection(pc)
                pos_proj_i = np.concatenate([pos_proj_range[np.newaxis, ...],
                                             pos_proj_pointcloud.transpose((2, 0, 1))], axis=0)
                pos_proj.append(pos_proj_i)
            for pc in neg_pc:
                neg_proj_pointcloud, neg_proj_range, _, _ = self.projection.doProjection(pc)
                neg_proj_i = np.concatenate([neg_proj_range[np.newaxis, ...],
                                             neg_proj_pointcloud.transpose((2, 0, 1))], axis=0)
                neg_proj.append(neg_proj_i)

        # do image augmentation
        if self.is_train and self.augmentor.do_img_aug:
            pc_ref, _, _ = self.dataset.loadDataByIndex(np.random.randint(len(self.dataset)) - 1, 1, 1)
            # pc_ref, _, _ = self.dataset.loadDataByIndex(1000, 1, 1)
            if self.depth_only:
                _, proj_range_ref, _, _ = self.projection.doProjection(pc_ref)
                proj_ref = proj_range_ref[np.newaxis, ...]
            else:
                proj_pointcloud_ref, proj_range_ref, proj_idx_ref, proj_mask_ref = self.projection.doProjection(pc_ref)
                proj_ref = np.concatenate([proj_range_ref[np.newaxis, ...], proj_pointcloud_ref.transpose((2, 0, 1))], axis=0)

            anchor_proj = self.augmentor.doAugmentationImage(anchor_proj, proj_ref)
            pos_img = []
            neg_img = []
            for pos_proj_i in pos_proj:
                pos_img.append(self.augmentor.doAugmentationImage(pos_proj_i, proj_ref))
            for neg_proj_i in neg_proj:
                neg_img.append(self.augmentor.doAugmentationImage(neg_proj_i, proj_ref))

        anchor_proj_tensor = torch.from_numpy(anchor_proj)
        pos_proj_tensor = torch.from_numpy(np.array(pos_img))
        neg_proj_tensor = torch.from_numpy(np.array(neg_img))

        if self.normalize:
            if self.depth_only:
                anchor_proj_tensor = (anchor_proj_tensor - self.proj_img_mean[0]) / self.proj_img_stds[0]
                pos_proj_tensor = (pos_proj_tensor - self.proj_img_mean[0]) / self.proj_img_stds[0]
                neg_proj_tensor = (neg_proj_tensor - self.proj_img_mean[0]) / self.proj_img_stds[0]
            else:
                mean = einops.repeat(self.proj_img_mean, 'c -> c h w', h=anchor_proj.shape[1], w=anchor_proj.shape[2])
                stds = einops.repeat(self.proj_img_stds, 'c -> c h w', h=anchor_proj.shape[1], w=anchor_proj.shape[2])
                anchor_proj_tensor = (anchor_proj_tensor - mean) / stds
                pos_proj_tensor = (pos_proj_tensor - mean) / stds
                neg_proj_tensor = (neg_proj_tensor - mean) / stds

        anchor_proj_tensor = anchor_proj_tensor.unsqueeze(0)
        return anchor_proj_tensor, pos_proj_tensor, neg_proj_tensor

    def __len__(self):
        if self.data_len > 0 and self.data_len < len(self.dataset):
            return self.data_len
        else:
            return len(self.dataset)
