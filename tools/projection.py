"""
The file copies from https://github.com/valeoai/rangevit/blob/main/dataset/preprocess/projection.py with trivial
modifications.
"""
import numpy as np


class RangeProjection(object):
    '''
    Project the 3D point cloud to 2D data with range projection

    Adapted from Z. Zhuang et al. https://github.com/ICEORY/PMF
    '''

    def __init__(self, fov_up=3.0, fov_down=-25.0, proj_w=1024, proj_h=64, fov_left=-180, fov_right=180, max_range=-1):
        # check params
        assert fov_up >= 0 and fov_down <= 0, f'require fov_up >= 0 and fov_down <= 0, fov_up/fov_down: {fov_up}/{fov_down}'
        assert fov_right >= 0 and fov_left <= 0, f'require fov_right >= 0 and fov_left <= 0, fov_right/fov_:left {fov_right}/{fov_left}'

        # params of fov angeles
        self.fov_up = fov_up / 180.0 * np.pi
        self.fov_down = fov_down / 180.0 * np.pi
        self.fov_v = abs(self.fov_up) + abs(self.fov_down)

        self.fov_left = fov_left / 180.0 * np.pi
        self.fov_right = fov_right / 180.0 * np.pi
        self.fov_horizon = abs(self.fov_left) + abs(self.fov_right)

        # params of proj img size
        self.proj_w = proj_w
        self.proj_h = proj_h

        self.cached_data = {}

        # min and max range if specify
        self.max_range = max_range
        if max_range > 0:
            self.min_range = 0.01


    def doProjection(self, pointcloud):
        """
        Project the 3D point cloud to 2D data with range projection
        Returns:
            proj_pointcloud: (numpy.ndarray) points used in the range image, in shape (proj_h, proj_w, channels)
            proj_range: (numpy.ndarray) range image, in shape (proj_h, proj_w)
            proj_idx: (numpy.ndarray) indices of points selected in original point cloud order, in shape (proj_h, proj_w)
            proj_mask: (numpy.ndarray) range image mask with no empty points, in shape (proj_h, proj_w)
        """
        # check input
        assert isinstance(pointcloud, np.ndarray), f'pointcloud should be numpy array, but got {type(pointcloud)}.'

        self.cached_data = {}
        # get depth of all points
        depth = np.linalg.norm(pointcloud[:, :3], 2, axis=1)
        if self.max_range > 0:
            pointcloud = pointcloud[(depth > self.min_range) & (depth < self.max_range)]
            depth = depth[(depth > self.min_range) & (depth < self.max_range)]

        # get point cloud components
        x = pointcloud[:, 0]
        y = pointcloud[:, 1]
        z = pointcloud[:, 2]

        # get angles of all points
        yaw = -np.arctan2(y, x)
        pitch = np.arcsin(z / (depth + 1e-8))

        # get projection in image coords
        proj_x = (yaw + abs(self.fov_left)) / self.fov_horizon      # normalized in [0, 1]
        # proj_x = 0.5 * (yaw / np.pi + 1.0)                          # normalized in [0, 1]
        proj_y = 1.0 - (pitch + abs(self.fov_down)) / self.fov_v    # normalized in [0, 1]

        # scale to image size using angular resolution
        proj_x *= self.proj_w
        proj_y *= self.proj_h

        px = np.maximum(np.minimum(self.proj_w, proj_x), 0)         # or proj_x.copy()
        py = np.maximum(np.minimum(self.proj_h, proj_y), 0)         # or proj_y.copy()

        # round and clamp for use as index
        proj_x = np.maximum(np.minimum(self.proj_w - 1, np.floor(proj_x)), 0).astype(np.int32)  # in [0, W-1]
        proj_y = np.maximum(np.minimum(self.proj_h - 1, np.floor(proj_y)), 0).astype(np.int32)  # in [0, H-1]

        # in default order
        self.cached_data['uproj_x_idx'] = proj_x.copy()
        self.cached_data['uproj_y_idx'] = proj_y.copy()
        self.cached_data['uproj_depth'] = depth.copy()
        self.cached_data['px'] = px
        self.cached_data['py'] = py

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        pointcloud = pointcloud[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # get projection result
        # range image
        proj_range = np.full((self.proj_h, self.proj_w), -1, dtype=np.float32)
        proj_range[proj_y, proj_x] = depth

        # point cloud in range image (x, y, z, intensity, ...)
        proj_pointcloud = np.full((self.proj_h, self.proj_w, pointcloud.shape[1]), -1, dtype=np.float32)
        proj_pointcloud[proj_y, proj_x] = pointcloud

        # range image selected points indices in original point cloud order
        proj_idx = np.full((self.proj_h, self.proj_w), -1, dtype=np.int32)
        proj_idx[proj_y, proj_x] = indices

        # range image mask with no empty points
        proj_mask = (proj_idx > 0).astype(np.int32)

        return proj_pointcloud, proj_range, proj_idx, proj_mask
