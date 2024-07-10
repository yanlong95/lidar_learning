"""
The file copies from https://github.com/valeoai/rangevit/blob/main/dataset/range_view_loader.py with modifications.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
import random


class AugmentParams(object):
    '''
    Adapted from Z. Zhuang et al. https://github.com/ICEORY/PMF
    '''

    def __init__(self, p_flipx=0., p_flipy=0., p_flipz=0.,
                 p_transx=0., trans_xmin=0., trans_xmax=0.,
                 p_transy=0., trans_ymin=0., trans_ymax=0.,
                 p_transz=0., trans_zmin=0., trans_zmax=0.,
                 p_rot_roll=0., rot_rollmin=0., rot_rollmax=0.,
                 p_rot_pitch=0., rot_pitchmin=0, rot_pitchmax=0.,
                 p_rot_yaw=0., rot_yawmin=0., rot_yawmax=0.,
                 yaw_only=True,
                 p_scale=0., scale_min=1.0, scale_max=1.0,
                 p_jitter=0., jitter_std=0.,
                 p_drop=0., drop_rmin=0., drop_rmax=0.,):
        self.p_flipx = p_flipx
        self.p_flipy = p_flipy
        self.p_flipz = p_flipz

        self.p_transx = p_transx
        self.trans_xmin = trans_xmin
        self.trans_xmax = trans_xmax

        self.p_transy = p_transy
        self.trans_ymin = trans_ymin
        self.trans_ymax = trans_ymax

        self.p_transz = p_transz
        self.trans_zmin = trans_zmin
        self.trans_zmax = trans_zmax

        self.p_rot_roll = p_rot_roll
        self.rot_rollmin = rot_rollmin
        self.rot_rollmax = rot_rollmax

        self.p_rot_pitch = p_rot_pitch
        self.rot_pitchmin = rot_pitchmin
        self.rot_pitchmax = rot_pitchmax

        self.p_rot_yaw = p_rot_yaw
        self.rot_yawmin = rot_yawmin
        self.rot_yawmax = rot_yawmax

        self.yaw_only = yaw_only

        self.p_scale = p_scale
        self.scale_min = scale_min
        self.scale_max = scale_max

        self.p_jitter = p_jitter
        self.jitter_std = jitter_std

        self.p_drop = p_drop
        self.drop_rmin = drop_rmin
        self.drop_rmax = drop_rmax

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

    def __str__(self):
        print('=== Augmentor parameters ===')
        print(f'p_flipx: {self.p_flipx}, p_flipy: {self.p_flipy}, p_flipyz: {self.p_flipz}')
        print(f'p_transx: {self.p_transx}, p_transxmin: {self.trans_xmin}, p_transxmax: {self.trans_xmax}')
        print(f'p_transy: {self.p_transy}, p_transymin: {self.trans_ymin}, p_transymax: {self.trans_ymax}')
        print(f'p_transz: {self.p_transz}, p_transzmin: {self.trans_zmin}, p_transzmax: {self.trans_zmax}')
        print(f'p_rotroll: {self.p_rot_roll}, rot_rollmin: {self.rot_rollmin}, rot_rollmax: {self.rot_rollmax}')
        print(f'p_rotpitch: {self.p_rot_pitch}, rot_pitchmin: {self.rot_pitchmin}, rot_pitchmax: {self.rot_pitchmax}')
        print(f'p_rotyaw: {self.p_rot_yaw}, rot_yawmin: {self.rot_yawmin}, rot_yawmax: {self.rot_yawmax}')
        print(f'p_scale: {self.p_scale}, scale_min: {self.scale_min}, scale_max: {self.scale_max}')
        print(f'p_jitter: {self.p_jitter}, jitter_std: {self.jitter_std}')
        print(f'p_drop: {self.p_drop}, drop_rmin: {self.drop_rmin}, drop_rmax: {self.drop_rmax}')
        print(f'yaw_only: {self.yaw_only}')

class Augmentor(object):
    def __init__(self, params: AugmentParams):
        self.parmas = params

    @staticmethod
    def flipX(pointcloud):
        """
        Flip the point cloud along x-axis.
        Args:
            pointcloud: (numpy array) point cloud in shape (n, channels).
        Returns:
            pointcloud: (numpy array) point cloud after flipping.
        """
        pointcloud[:, 0] = -pointcloud[:, 0]
        return pointcloud

    @staticmethod
    def flipY(pointcloud):
        """
        Flip the point cloud along y-axis.
        Args:
            pointcloud: (numpy array) point cloud in shape (n, channels).
        Returns:
            pointcloud: (numpy array) point cloud after flipping.
        """
        pointcloud[:, 1] = -pointcloud[:, 1]
        return pointcloud

    @staticmethod
    def translation(pointcloud, x, y, z):
        """
        Translate the point cloud.
        Args:
            pointcloud: (numpy array) point cloud in shape (n, channels).
            x: (float) translation along x-axis.
            y: (float) translation along y-axis.
            z: (float) translation along z-axis.
        Returns:
            pointcloud: (numpy array) point cloud after translation.
        """
        pointcloud[:, 0] += x
        pointcloud[:, 1] += y
        pointcloud[:, 2] += z
        return pointcloud

    @staticmethod
    def rotation(pointcloud, roll, pitch, yaw, degrees=True):
        """
        Rotate the point cloud.
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
        pointcloud[:, :3] = pointcloud[:, :3] @ rot_matrix.T
        return pointcloud

    @staticmethod
    def randomRotation(pointcloud, yaw_only=True):
        """
        Randomly rotate the point cloud.
        Args:
            pointcloud: (numpy array) point cloud in shape (n, channels).
            yaw_only: (bool) whether to rotate only around z-axis.
        Returns:
            pointcloud: (numpy array) point cloud after rotation.
        """
        if not yaw_only:
            rot_matrix = R.random().as_matrix()
        else:
            yaw_angle = np.random.random() * 360
            rot_matrix = R.from_euler('z', yaw_angle, degrees=True).as_matrix()

        pointcloud[:, :3] = pointcloud[:, :3] @ rot_matrix.T
        return pointcloud

    # ------------------------ new added (modified) ------------------------
    @staticmethod
    def flipZ(pointcloud):
        """
        Flip the point cloud along z-axis.
        Args:
            pointcloud: (numpy array) point cloud in shape (n, channels).
        Returns:
            pointcloud: (numpy array) point cloud after flipping.
        """
        pointcloud[:, 2] = -pointcloud[:, 2]
        return pointcloud

    @staticmethod
    def scale_cloud(pointcloud, scale_min, scale_max):
        """
        Randomly scale the point cloud (only in x, y axis, e.g. move objects further).
        //TODO: add default scale factors.
        Args:
            pointcloud: (numpy array) point cloud in shape (n, channels).
            scale_min: (float) minimum scale factor.
            scale_max: (float) maximum scale factor.
        Returns:
            pointcloud: (numpy array) point cloud after scaling.
        """
        scale = np.random.uniform(scale_min, scale_max)
        pointcloud[:, :2] *= scale
        return pointcloud

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
        pointcloud[:, :3] += jitter
        return pointcloud

    @staticmethod
    def randomDrop(pointcloud, drop_rmin, drop_rmax):
        """
        Randomly drop points from the point cloud.
        Args:
            pointcloud: (numpy array) point cloud in shape (n, channels).
            drop_rmin: (float) minimum drop rate.
            drop_rmax: (float) maximum drop rate.
        Returns:
            pointcloud: (numpy array) point cloud after dropping points.
        """
        min_num_drop = int(len(pointcloud) * drop_rmin)
        max_num_drop = int(len(pointcloud) * drop_rmax)
        drop_size = np.random.randint(min_num_drop, max_num_drop)
        drop_points = np.unique(np.random.randint(low=0, high=len(pointcloud)-1, size=drop_size))
        pointcloud = np.delete(pointcloud, drop_points, axis=0)
        return pointcloud

    def doAugmentation(self, pointcloud):
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
            trans_x = random.uniform(
                self.parmas.trans_xmin, self.parmas.trans_xmax)
        else:
            trans_x = 0
        rand = random.uniform(0, 1)
        if rand < self.parmas.p_transy:
            trans_y = random.uniform(
                self.parmas.trans_ymin, self.parmas.trans_ymax)
        else:
            trans_y = 0
        rand = random.uniform(0, 1)
        if rand < self.parmas.p_transz:
            trans_z = random.uniform(
                self.parmas.trans_zmin, self.parmas.trans_zmax)
        else:
            trans_z = 0
        pointcloud = self.translation(pointcloud, trans_x, trans_y, trans_z)

        # rotation
        rand = random.uniform(0, 1)
        if rand < self.parmas.p_rot_roll:
            rot_roll = random.uniform(
                self.parmas.rot_rollmin, self.parmas.rot_rollmax)
        else:
            rot_roll = 0
        rand = random.uniform(0, 1)
        if rand < self.parmas.p_rot_pitch:
            rot_pitch = random.uniform(
                self.parmas.rot_pitchmin, self.parmas.rot_pitchmax)
        else:
            rot_pitch = 0
        rand = random.uniform(0, 1)
        if rand < self.parmas.p_rot_yaw:
            rot_yaw = random.uniform(
                self.parmas.rot_yawmin, self.parmas.rot_yawmax)
        else:
            rot_yaw = 0
        pointcloud = self.rotation(pointcloud, rot_roll, rot_pitch, rot_yaw)

        # ------------------------ new added ------------------------
        # jitter
        rand = random.uniform(0, 1)
        if rand < self.parmas.p_jitter:
            pointcloud = self.randomJitter(pointcloud, self.parmas.jitter_std)

        # flip z
        rand = random.uniform(0, 1)
        if rand < self.parmas.p_flipz:
            pointcloud = self.flipZ(pointcloud)

        # drop
        rand = random.uniform(0, 1)
        if rand < self.parmas.p_drop:
            pointcloud = self.randomDrop(pointcloud, self.parmas.drop_rmin, self.parmas.drop_rmax)

        return pointcloud
