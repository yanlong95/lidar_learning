"""
The file copies from https://github.com/valeoai/rangevit/blob/main/dataset/range_view_loader.py with modifications.
"""
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R
import random
from dataclasses import dataclass, field, fields
from typing import List


@dataclass
class AugmentParams:
    p_flipx: float = 0.
    p_flipy: float = 0.
    p_flipz: float = 0.

    p_transx: float = 0.
    trans_xmin: float = 0.
    trans_xmax: float = 0.

    p_transy: float = 0.
    trans_ymin: float = 0.
    trans_ymax: float = 0.

    p_transz: float = 0.
    trans_zmin: float = 0.
    trans_zmax: float = 0.

    p_rot_roll: float = 0.
    rot_rollmin: float = 0.
    rot_rollmax: float = 0.

    p_rot_pitch: float = 0.
    rot_pitchmin: float = 0.
    rot_pitchmax: float = 0.

    p_rot_yaw: float = 0.
    rot_yawmin: float = 0.
    rot_yawmax: float = 0.

    # ------------------------ new added ------------------------
    p_scale: float = 0.
    scale_min: float = 1.0
    scale_max: float = 1.0

    p_jitter: float = 0.
    jitter_std: float = 0.

    p_drop: float = 0.
    drop_min: float = 0.
    drop_max: float = 0.

    p_range_mix: float = 0.
    k_mix: List[int] = field(default_factory=list)

    p_range_union: float = 0.
    k_union: float = 0.

    p_range_shift: float = 0.
    range_shift_min: float = 0.
    range_shift_max: float = 0.

    p_range_drop: float = 0.
    range_drop_min: float = 0.
    range_drop_max: float = 0.

    p_range_hflip: float = 0.
    # -----------------------------------------------------------

    def setScaleParams(self, p_scale, scale_min, scale_max):
        self.p_scale = p_scale
        self.scale_min = scale_min
        self.scale_max = scale_max

    def setJitterParams(self, p_jitter, jitter_std):
        self.p_jitter = p_jitter
        self.jitter_std = jitter_std

    def setDropParams(self, p_drop, drop_min, drop_max):
        self.p_drop = p_drop
        self.drop_min = drop_min
        self.drop_max = drop_max

    def setFlipProb(self, p_flipx, p_flipy, p_flipz):
        self.p_flipx = p_flipx
        self.p_flipy = p_flipy
        self.p_flipz = p_flipz

    def setTranslationParams(self,
                           p_transx=0., trans_xmin=0., trans_xmax=0.,
                           p_transy=0., trans_ymin=0., trans_ymax=0.,
                           p_transz=0., trans_zmin=0., trans_zmax=0.):
        self.p_transx = p_transx
        self.trans_xmin = trans_xmin
        self.trans_xmax = trans_xmax

        self.p_transy = p_transy
        self.trans_ymin = trans_ymin
        self.trans_ymax = trans_ymax

        self.p_transz = p_transz
        self.trans_zmin = trans_zmin
        self.trans_zmax = trans_zmax

    def setRotationParams(self,
                        p_rot_roll=0., rot_rollmin=0., rot_rollmax=0.,
                        p_rot_pitch=0., rot_pitchmin=0, rot_pitchmax=0.,
                        p_rot_yaw=0., rot_yawmin=0., rot_yawmax=0.):

        self.p_rot_roll = p_rot_roll
        self.rot_rollmin = rot_rollmin
        self.rot_rollmax = rot_rollmax

        self.p_rot_pitch = p_rot_pitch
        self.rot_pitchmin = rot_pitchmin
        self.rot_pitchmax = rot_pitchmax

        self.p_rot_yaw = p_rot_yaw
        self.rot_yawmin = rot_yawmin
        self.rot_yawmax = rot_yawmax

    def setRangeMixParams(self, p_range_mix, k_mix):
        self.p_range_mix = p_range_mix
        self.k_mix = k_mix

    def setRangeUnionParams(self, p_range_union, k_union):
        self.p_range_union = p_range_union
        self.k_union = k_union

    def setRangeShiftParams(self, p_range_shift, range_shift_min, range_shift_max):
        self.p_range_shift = p_range_shift
        self.range_shift_min = range_shift_min
        self.range_shift_max = range_shift_max

    def setRangeDropParams(self, p_range_drop, range_drop_min, range_drop_max):
        self.p_range_drop = p_range_drop
        self.range_drop_min = range_drop_min
        self.range_drop_max = range_drop_max

    def setRangeHFlipParams(self, p_range_hflip):
        self.p_range_hflip = p_range_hflip

    def __str__(self):
        params = '=== Augmentor parameters ===\n'                                                     
        for item in vars(self).items():
            params += f'{item[0]}: {item[1]}\n'
        return params


class Augmentor(object):
    def __init__(self, params: AugmentParams, do_pc_aug=True, do_img_aug=True, do_img_crop=False):
        self.parmas = params
        self.do_pc_aug = do_pc_aug
        self.do_img_aug = do_img_aug
        self.do_img_crop = do_img_crop

    @staticmethod
    def flipX(pointcloud):
        """
        Flip the point cloud along x-axis (equal to range image flip + shift).
        Args:
            pointcloud: (numpy array) point cloud in shape (n, channels).
        Returns:
            pointcloud: (numpy array) point cloud after flipping.
        """
        pointcloud_ = pointcloud.copy()
        pointcloud_[:, 0] = -pointcloud_[:, 0]
        return pointcloud_

    @staticmethod
    def flipY(pointcloud):
        """
        Flip the point cloud along y-axis (equal to range image flip).
        Args:
            pointcloud: (numpy array) point cloud in shape (n, channels).
        Returns:
            pointcloud: (numpy array) point cloud after flipping.
        """
        pointcloud_ = pointcloud.copy()
        pointcloud_[:, 1] = -pointcloud_[:, 1]
        return pointcloud_

    @staticmethod
    def translation(pointcloud, x, y, z):
        """
        Translate the point cloud (torsion in range image NOT RECOMMEND).
        Args:
            pointcloud: (numpy array) point cloud in shape (n, channels).
            x: (float) translation along x-axis.
            y: (float) translation along y-axis.
            z: (float) translation along z-axis.
        Returns:
            pointcloud: (numpy array) point cloud after translation.
        """
        pointcloud_ = pointcloud.copy()
        pointcloud_[:, 0] += x
        pointcloud_[:, 1] += y
        pointcloud_[:, 2] += z
        return pointcloud_

    @staticmethod
    def rotation(pointcloud, roll, pitch, yaw, degrees=True):
        """
        Rotate the point cloud (roll, pitch are torsion, yaw is shifting in range image ROLL AND PITCH SHOULD BE SMALL).
        Args:
            pointcloud: (numpy array) point cloud in shape (n, channels).
            roll: (float) rotation angle around x-axis.
            pitch: (float) rotation angle around y-axis.
            yaw: (float) rotation angle around z-axis.
            degrees: (bool) whether the input angles are in degrees.
        Returns:
            pointcloud: (numpy array) point cloud after rotation.
        """
        rot_matrix = R.from_euler('zyx', [yaw, pitch, roll], degrees=degrees).as_matrix()
        pointcloud_ = pointcloud.copy()
        pointcloud_[:, :3] = pointcloud_[:, :3] @ rot_matrix.T
        return pointcloud_

    @staticmethod
    def randomRotation(pointcloud):
        """
        Randomly rotate the point cloud.
        Args:
            pointcloud: (numpy array) point cloud in shape (n, channels).
        Returns:
            pointcloud: (numpy array) point cloud after rotation.
        """
        yaw_angle = np.random.random() * 360
        rot_matrix = R.from_euler('z', yaw_angle, degrees=True).as_matrix()

        pointcloud_ = pointcloud.copy()
        pointcloud_[:, :3] = pointcloud_[:, :3] @ rot_matrix.T
        return pointcloud_

    # ------------------------ new added (modified) ------------------------
    @staticmethod
    def flipZ(pointcloud):
        """
        Flip the point cloud along z-axis (flip and shifting in vertical direction in range image NOT RECOMMEND).
        Args:
            pointcloud: (numpy array) point cloud in shape (n, channels).
        Returns:
            pointcloud: (numpy array) point cloud after flipping.
        """
        pointcloud_ = pointcloud.copy()
        pointcloud_[:, 2] = -pointcloud_[:, 2]
        return pointcloud_

    @staticmethod
    def scale_cloud(pointcloud, scale_min, scale_max):
        """
        Randomly scale the point cloud (only in x, y axis. Zoom in and out in range image).
        //TODO: add default scale factors.
        Args:
            pointcloud: (numpy array) point cloud in shape (n, channels).
            scale_min: (float) minimum scale factor.
            scale_max: (float) maximum scale factor.
        Returns:
            pointcloud: (numpy array) point cloud after scaling.
        """
        scale = np.random.uniform(scale_min, scale_max)
        pointcloud_ = pointcloud.copy()
        pointcloud_[:, :2] *= scale
        return pointcloud_

    @staticmethod
    def randomJitter(pointcloud, jitter_std):
        """
        Randomly jitter points in the point cloud.
        Args:
            pointcloud: (numpy array) point cloud in shape (n, channels).
            jitter_std: (float) standard deviation of the normal jitter distribution.
        Returns:
            pointcloud: (numpy array) point cloud after jittering.
        """
        jitter = np.clip(np.random.normal(0, jitter_std, (pointcloud.shape[0], 3)), -3*jitter_std, 3*jitter_std)
        pointcloud_ = pointcloud.copy()
        pointcloud_[:, :3] += jitter
        return pointcloud_

    @staticmethod
    def randomDrop(pointcloud, drop_min, drop_max):
        """
        Randomly drop points from the point cloud.
        Args:
            pointcloud: (numpy array) point cloud in shape (n, channels).
            drop_min: (float) minimum drop rate.
            drop_max: (float) maximum drop rate.
        Returns:
            pointcloud: (numpy array) point cloud after dropping points.
        """
        min_num_drop = int(len(pointcloud) * drop_min)
        max_num_drop = int(len(pointcloud) * drop_max)
        drop_size = np.random.randint(min_num_drop, max_num_drop)
        drop_points = np.unique(np.random.randint(low=0, high=len(pointcloud)-1, size=drop_size))
        pointcloud = np.delete(pointcloud, drop_points, axis=0)
        return pointcloud

    @staticmethod
    def rangeMix(image1, image2, k_mix):
        """
        Switch certain rows between two range images.
        !!!
        Note, the original paper switches a big chunk of pixels at once. The detail is not provided. Accord the paper
        description, I randomly switch k continuous rows to switch for each time.
        !!!
        Args:
            image1: (numpy array) range image in shape (channels, height, width) or (height, width).
            image2: (numpy array) range image in shape (channels, height, width) or (height, width).
            k_mix: (list) list of number of rows to switch, randomly select one each time.
        Returns:
            image: (numpy array) range image after switching rows.
        """
        if len(image1.shape) == 3:
            height = image1.shape[1]
            num_mix = np.random.choice(k_mix)
            row_idx = np.random.randint(0, height - num_mix)
            image1_ = image1.copy()
            image1_[:, row_idx:row_idx+num_mix, :] = image2[:, row_idx:row_idx+num_mix, :]
        else:
            height = image1.shape[0]
            num_mix = np.random.choice(k_mix)
            row_idx = np.random.randint(0, height - num_mix)
            image1_ = image1.copy()
            image1_[row_idx:row_idx+num_mix, :] = image2[row_idx:row_idx+num_mix, :]
        return image1_

    @staticmethod
    def rangeUnion(image1, image2, k_union):
        """
        Union two range images.
        Args:
            image1: (numpy array) range image in shape (channels, height, width) or (height, width).
            image2: (numpy array) range image in shape (channels, height, width) or (height, width).
            k_union: (float) percent of empty pixels in image1 to fill with image2.
        Returns:
            image: (numpy array) range image after union.
        """
        if len(image1.shape) == 3:
            mask = image1[0, :, :] < 0      # assume 1st channel is depth
            mask *= np.random.random(mask.shape) > k_union
            image1_ = image1.copy()
            image1_[:, mask] = image2[:, mask]
        else:
            mask = image1 < 0
            mask *= np.random.random(mask.shape) > k_union
            image1_ = image1.copy()
            image1_[mask] = image2[mask]
        return image1_

    @staticmethod
    def rangeShift(image, shift_min, shift_max):
        """
        Shift the range image along width (yaw rotation).
        Args:
            image: (numpy array) range image in shape (channels, height, width) or (height, width).
            shift_min: (float) minimum shift ratio.
            shift_max: (float) maximum shift ratio.
        Returns:
            image: (numpy array) range image after shifting.
        """
        if len(image.shape) == 3:
            width = image.shape[2]
            shift = np.random.randint(int(shift_min * width), int(shift_max * width))
            image_ = image.copy()
            image_ = np.concatenate((image_[:, :, shift:], image_[:, :, :shift]), axis=2)
        else:
            width = image.shape[1]
            shift = np.random.randint(int(shift_min * width), int(shift_max * width))
            image_ = image.copy()
            image_ = np.concatenate((image_[:, shift:], image_[:, :shift]), axis=1)
        return image_

    @staticmethod
    def rangeDrop(image, drop_min, drop_max):
        """
        Randomly drop pixels from the range image. Pixels with small depth are more likely to be dropped.
        Args:
            image: (numpy array) range image in shape (channels, height, width) or (height, width).
            drop_min: (float) minimum drop rate.
            drop_max: (float) maximum drop rate.
        Returns:
            image: (numpy array) range image after dropping pixels.
        """
        if len(image.shape) == 3:
            drop_rate = np.random.uniform(drop_min, drop_max)
            pixel_drop_ratio = np.ones_like(image[0, :, :]) - image[0, :, :] / np.max(image[0, :, :])
            pixel_drop_ratio = 0.8 * pixel_drop_ratio + 0.2
            mask = np.random.rand(image.shape[1], image.shape[2]) < drop_rate * pixel_drop_ratio
            image_ = image.copy()
            image_[:, mask] = -1
        else:
            drop_rate = np.random.uniform(drop_min, drop_max)
            pixel_drop_ratio = np.ones_like(image) - image / np.max(image)
            pixel_drop_ratio = 0.98 * pixel_drop_ratio + 0.02
            mask = np.random.rand(image.shape[0], image.shape[1]) < drop_rate * pixel_drop_ratio
            image_ = image.copy()
            image_[mask] = -1
        return image_

    @staticmethod
    def rangeHFlip(image):
        """
        Horizontally flip the range image.
        Args:
            image: (numpy array) range image in shape (channels, height, width) or (height, width).
        Returns:
            image: (numpy array) range image after flipping.
        """
        image_ = np.flip(image, axis=-1)
        return image_

    def doAugmentationPointcloud(self, pointcloud):
        # flip augment
        rand = random.uniform(0, 1)
        if rand < self.parmas.p_flipx:
            pointcloud = self.flipX(pointcloud)

        rand = random.uniform(0, 1)
        if rand < self.parmas.p_flipy:
            pointcloud = self.flipY(pointcloud)

        # scale augment
        rand = random.uniform(0, 1)
        if rand < self.parmas.p_scale:
            pointcloud = self.scale_cloud(pointcloud, self.parmas.scale_min, self.parmas.scale_max)

        # translation
        rand = random.uniform(0, 1)
        if rand < self.parmas.p_transx:
            trans_x = random.uniform(self.parmas.trans_xmin, self.parmas.trans_xmax)
        else:
            trans_x = 0
        rand = random.uniform(0, 1)
        if rand < self.parmas.p_transy:
            trans_y = random.uniform(self.parmas.trans_ymin, self.parmas.trans_ymax)
        else:
            trans_y = 0
        rand = random.uniform(0, 1)
        if rand < self.parmas.p_transz:
            trans_z = random.uniform(self.parmas.trans_zmin, self.parmas.trans_zmax)
        else:
            trans_z = 0
        pointcloud = self.translation(pointcloud, trans_x, trans_y, trans_z)

        # rotation
        rand = random.uniform(0, 1)
        if rand < self.parmas.p_rot_roll:
            rot_roll = random.uniform(self.parmas.rot_rollmin, self.parmas.rot_rollmax)
        else:
            rot_roll = 0
        rand = random.uniform(0, 1)
        if rand < self.parmas.p_rot_pitch:
            rot_pitch = random.uniform(self.parmas.rot_pitchmin, self.parmas.rot_pitchmax)
        else:
            rot_pitch = 0
        rand = random.uniform(0, 1)
        if rand < self.parmas.p_rot_yaw:
            rot_yaw = random.uniform(self.parmas.rot_yawmin, self.parmas.rot_yawmax)
        else:
            rot_yaw = 0
        pointcloud = self.rotation(pointcloud, rot_roll, rot_pitch, rot_yaw)

        # ------------------------ new added ------------------------
        # flip z
        rand = random.uniform(0, 1)
        if rand < self.parmas.p_flipz:
            pointcloud = self.flipZ(pointcloud)

        # jitter
        rand = random.uniform(0, 1)
        if rand < self.parmas.p_jitter:
            pointcloud = self.randomJitter(pointcloud, self.parmas.jitter_std)

        # drop
        rand = random.uniform(0, 1)
        if rand < self.parmas.p_drop:
            pointcloud = self.randomDrop(pointcloud, self.parmas.drop_min, self.parmas.drop_max)

        return pointcloud

    def doAugmentationImage(self, pointcloud, pointcloud_ref):
        # range mix
        rand = random.uniform(0, 1)
        if rand < self.parmas.p_range_mix:
            pointcloud = self.rangeMix(pointcloud, pointcloud_ref, self.parmas.k_mix)

        # range union
        rand = random.uniform(0, 1)
        if rand < self.parmas.p_range_union:
            pointcloud = self.rangeUnion(pointcloud, pointcloud_ref, self.parmas.k_union)

        # range shift
        rand = random.uniform(0, 1)
        if rand < self.parmas.p_range_shift:
            pointcloud = self.rangeShift(pointcloud, self.parmas.range_shift_min, self.parmas.range_shift_max)

        # range drop
        rand = random.uniform(0, 1)
        if rand < self.parmas.p_range_drop:
            pointcloud = self.rangeDrop(pointcloud, self.parmas.range_drop_min, self.parmas.range_drop_max)

        # range hflip
        rand = random.uniform(0, 1)
        if rand < self.parmas.p_range_hflip:
            pointcloud = self.rangeHFlip(pointcloud)

        return pointcloud


if __name__ == '__main__':
    from projection import RangeProjection
    # test augmentation
    config_path = '/home/vectr/PycharmProjects/lidar_learning/data/kitti/dataset/config_kitti.yml'
    config_path = '/Users/yanlong/PycharmProjects/lidar_learning/data/kitti/dataset/config_kitti.yml'
    config = yaml.safe_load(open(config_path))
    augment_pc_config = config['augmentation_pointcloud']
    augment_img_config = config['augmentation_image']
    projection_config = config['sensor']

    # assign values to augment_params
    augment_params = AugmentParams()
    field_name = {f.name for f in fields(AugmentParams) if f.init}
    for key, value in augment_pc_config.items():
        if key in field_name:
            setattr(augment_params, key, value)

    for key, value in augment_img_config.items():
        if key in field_name:
            setattr(augment_params, key, value)

    augmentor = Augmentor(augment_params)
    projector = RangeProjection()

    pc1 = '/Volumes/T7/Datasets/public_datasets/kitti/dataset/sequences/00/velodyne/001000.bin'
    pc2 = '/Volumes/T7/Datasets/public_datasets/kitti/dataset/sequences/00/velodyne/002000.bin'

    # load point cloud
    from tools.fileloader import read_pc
    import matplotlib.pyplot as plt
    import time
    pc1 = read_pc(pc1)
    pc2 = read_pc(pc2)

    # check augmentation functions
    t0 = time.time()
    pc1_aug = augmentor.randomJitter(pc1, 0.03)
    t1 = time.time()

    _, img1_aug, _, _ = projector.doProjection(pc1_aug)
    _, img1, _, _ = projector.doProjection(pc1)
    _, img2, _, _ = projector.doProjection(pc2)
    t2 = time.time()

    img1_aug = augmentor.doAugmentationImage(img1_aug, img2)
    t3 = time.time()

    print(f'PC augmentation time: {t1-t0}')
    print(f'Avg projection time: {(t2-t1) / 3}')
    print(f'Img augmentation time: {t3-t2}')

    fig, [ax1, ax2, ax3] = plt.subplots(3, 1)
    ax1.imshow(img1)
    ax2.imshow(img2)
    ax3.imshow(img1_aug)
    plt.show()
