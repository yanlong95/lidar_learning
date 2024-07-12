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
    def __init__(self, dataset, config, data_len=-1, is_train=True, return_uproj=False):
        self.dataset = dataset
        self.config = config
        self.is_train = is_train
        self.data_len = data_len
        self.return_uproj = return_uproj

        augment_params = AugmentParams()
        augment_pc_config = self.config['augmentation_pointcloud']
        augment_img_config = self.config['augmentation_image']
        projection_config = self.config['sensor']

        # Point cloud augmentations and image augmentations
        if self.is_train:
            augment_params = AugmentParams()
            field_name = {f.name for f in fields(AugmentParams) if f.init}
            for key, value in augment_pc_config.items():
                if key in field_name:
                    setattr(augment_params, key, value)

            for key, value in augment_img_config.items():
                if key in field_name:
                    setattr(augment_params, key, value)

            self.augmentor = Augmentor(augment_params)
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
        self.image_crop = augment_img_config['image_crop']
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

        self.proj_p_hflip = augment_img_config.get('p_hflip', 0.0)
        if self.proj_p_hflip > 0.0:
            print(f'Horizontal flip of range projections with p={self.proj_p_hflip}')


    def __getitem__(self, index):
        '''
        proj_feature_tensor: CxHxW
        proj_sem_label_tensor: HxW
        proj_mask_tensor: HxW
        '''
        pointcloud, sem_label, inst_label = self.dataset.loadDataByIndex(index)
        if self.is_train:
            pointcloud = self.augmentor.doAugmentationPointcloud(pointcloud)  # n, 4
        proj_pointcloud, proj_range, proj_idx, proj_mask = self.projection.doProjection(pointcloud)

        proj_mask_tensor = torch.from_numpy(proj_mask)
        mask = proj_idx > 0
        proj_sem_label = np.zeros((proj_mask.shape[0], proj_mask.shape[1]), dtype=np.float32)
        proj_sem_label[mask] = self.dataset.labelMapping(sem_label[proj_idx[mask]])
        proj_sem_label_tensor = torch.from_numpy(proj_sem_label)
        proj_sem_label_tensor = proj_sem_label_tensor * proj_mask_tensor.float()

        proj_range_tensor = torch.from_numpy(proj_range)
        proj_xyz_tensor = torch.from_numpy(proj_pointcloud[..., :3])
        proj_intensity_tensor = torch.from_numpy(proj_pointcloud[..., 3])
        proj_intensity_tensor = proj_intensity_tensor.ne(-1).float() * proj_intensity_tensor
        proj_feature_tensor = torch.cat(
            [proj_range_tensor.unsqueeze(0), proj_xyz_tensor.permute(2, 0, 1), proj_intensity_tensor.unsqueeze(0)], 0)

        proj_feature_tensor = (proj_feature_tensor - self.proj_img_mean[:, None, None]) / self.proj_img_stds[:, None,
                                                                                          None]
        proj_feature_tensor = proj_feature_tensor * proj_mask_tensor.unsqueeze(0).float()

        if self.return_uproj:
            sem_label = self.dataset.labelMapping(sem_label)
            sem_label = torch.from_numpy(sem_label).long()

            uproj_x_tensor = torch.from_numpy(self.projection.cached_data['uproj_x_idx']).long()
            uproj_y_tensor = torch.from_numpy(self.projection.cached_data['uproj_y_idx']).long()
            uproj_depth_tensor = torch.from_numpy(self.projection.cached_data['uproj_depth']).float()

            return proj_feature_tensor, proj_sem_label_tensor, proj_mask_tensor, torch.from_numpy(
                proj_range), uproj_x_tensor, uproj_y_tensor, uproj_depth_tensor, sem_label
        else:
            proj_tensor = torch.cat(
                (proj_feature_tensor,
                proj_sem_label_tensor.unsqueeze(0),
                proj_mask_tensor.float().unsqueeze(0)), dim=0)

            # Data augmentation
            proj_tensor = self.aug_ops(proj_tensor)

            return proj_tensor[0:5], proj_tensor[5], proj_tensor[6]

    def __len__(self):
        if self.data_len > 0 and self.data_len < len(self.dataset):
            return self.data_len
        else:
            return len(self.dataset)


def count_num_of_valid_points(py, px, offset_y, offset_x, h, w):
    py = (py - offset_y) / h
    px = (px - offset_x) / w
    valid = (px >= 0) & (px <= 1) & (py >= 0) & (py <= 1)
    return valid.astype('float64').sum()


def crop_inputs(proj_tensor, px, py, points_xyz, labels, crop_size, center_crop=False, p_hflip=0.0):
    if center_crop:
        _, h, w = proj_tensor.shape
        assert h == crop_size[0] and w == crop_size[1]
        offset_y, offset_x = 0, 0
    else:
        MIN_NUM_POINTS = 1
        NUM_ITERS = 10
        for _ in range(NUM_ITERS):
            offset_y, offset_x, h, w = T.RandomCrop.get_params(proj_tensor, crop_size)
            num_valid_points = count_num_of_valid_points(py, px, offset_y, offset_x, h, w)
            if num_valid_points > MIN_NUM_POINTS:
                break
            print(f'num_valid_points = {num_valid_points}')
        assert h == crop_size[0] and w == crop_size[1]
    proj_tensor = TF.crop(proj_tensor, offset_y, offset_x, h, w)

    py = (py - offset_y) / h
    px = (px - offset_x) / w
    valid = (px >= 0) & (px <= 1) & (py >= 0) & (py <= 1)

    labels = labels[valid]
    px = px[valid]
    py = py[valid]
    points_xyz = points_xyz[valid, :]
    px = 2.0 * (px - 0.5)
    py = 2.0 * (py - 0.5)

    if np.random.uniform() < p_hflip:
        proj_tensor = TF.hflip(proj_tensor)
        px *= -1

    return proj_tensor, px, py, points_xyz, labels
