import os
import sys
import yaml
import torch
import faiss
import tqdm
import numpy as np
import open3d as o3d
import cv2


def load_keyframes_descriptors():
    pass


def compute_range_image(points, params, proj_W=900, proj_H=32):
    # lidar view parameters
    fov_up = params['fov_up'] / 180.0 * np.pi  		# field of view up in radians
    fov_down = params['fov_down'] / 180.0 * np.pi  	# field of view down in radians
    fov = abs(fov_down) + abs(fov_up)  				# get field of view total in radians

    # get depth of all points (filter the points 0 and out of the maximal range)
    depth = np.linalg.norm(points[:, :3], axis=1)
    points = points[(depth > params['eps']) & (depth < params['max_range_0.8'])]
    depth = depth[(depth > params['eps']) & (depth < params['max_range_0.8'])]

    # get scan components
    points_x = points[:, 0]
    points_y = points[:, 1]
    points_z = points[:, 2]

    # get angles of all points
    yaw = -np.arctan2(points_y, points_x)
    pitch = np.arcsin(points_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

    # order in decreasing depth
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    proj_range = -np.ones((proj_H, proj_W), dtype=np.float32)	# [H,W] range (-1 is no data)
    proj_range[proj_y, proj_x] = depth
    return proj_range


def compute_range_image_tensor(points, params, proj_W=900, proj_H=32, device='cuda'):
    # lidar view parameters
    fov_up = params['fov_up'] / 180.0 * np.pi  		# field of view up in radians
    fov_down = params['fov_down'] / 180.0 * np.pi  	# field of view down in radians
    fov = abs(fov_down) + abs(fov_up)  				# get field of view total in radians

    # get depth of all points (filter the points 0 and out of the maximal range)
    depth = torch.linalg.norm(points[:, :3], axis=1).type(torch.float32)
    points = points[(depth > params['eps']) & (depth < params['max_range_0.8'])]
    depth = depth[(depth > params['eps']) & (depth < params['max_range_0.8'])]

    # get scan components
    points_x = points[:, 0]
    points_y = points[:, 1]
    points_z = points[:, 2]

    # get angles of all points
    yaw = -torch.arctan2(points_y, points_x)
    pitch = torch.arcsin(points_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / torch.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]

    # round and clamp for use as index
    proj_x = torch.floor(proj_x)
    proj_x = torch.minimum((proj_W - 1)*torch.ones_like(proj_x), proj_x)
    proj_x = torch.maximum(torch.zeros_like(proj_x), proj_x).type(torch.int32)  # in [0,W-1]

    proj_y = torch.floor(proj_y)
    proj_y = torch.minimum((proj_H - 1)*torch.ones_like(proj_y), proj_y)
    proj_y = torch.maximum(torch.zeros_like(proj_y), proj_y).type(torch.int32)  # in [0,H-1]

    # order in decreasing depth
    order = torch.argsort(depth, descending=True)
    depth = depth[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    proj_range = -torch.ones((proj_H, proj_W), dtype=torch.float32).to(device)	# [H,W] range (-1 is no data)
    proj_range[proj_y, proj_x] = depth
    return proj_range


def compute_descriptor(model, image, device):
    with torch.no_grad():
        if torch.is_tensor(image):
            image_tensor = image.to(device)
        else:
            image_tensor = torch.from_numpy(image).type(torch.float32).to(device)

        image_tensor = torch.unsqueeze(image_tensor, dim=0)
        image_tensor = torch.unsqueeze(image_tensor, dim=0)
        image_tensor = torch.cat((image_tensor, image_tensor), dim=0)			# might delete later

        model.eval()
        image_descriptor = model(image_tensor)
        image_descriptor = image_descriptor[0, :].detach().cpu().numpy().astype('float32')
        return image_descriptor


def compute_top_n_keyframes(keyframes, frame, top_n=5, device='cuda'):
    top_n = min(top_n, len(keyframes))
    index = faiss.IndexFlatL2(frame.shape[0])
    index.add(keyframes)
    _, I = index.search(frame.reshape(1, -1), top_n)
    print(I.shape)




if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    # import time
    #
    # from tools.fileloader import load_xyz_rot, read_pc, load_files
    # keyframe_poses, keyframe_rot = load_xyz_rot('/media/vectr/vectr3/Dataset/arl/poses/parking-lot/poses_kf.txt')
    # # plt.scatter(keyframe_poses[:,0], keyframe_poses[:,1])
    # # plt.show()
    #
    # keyframes_pcd_path = load_files('/media/vectr/vectr3/Dataset/arl/pcd_files/parking-lot_keyframes')
    # keyframe2_path = keyframes_pcd_path[2]
    # keyframe0_path = keyframes_pcd_path[0]
    # keyframe1_path = keyframes_pcd_path[1]
    # keyframe3_path = keyframes_pcd_path[3]
    # keyframe9_path = keyframes_pcd_path[-1]
    #
    # keyframe2 = read_pc(keyframe2_path)
    # keyframe0 = read_pc(keyframe0_path)
    # keyframe1 = read_pc(keyframe1_path)
    # keyframe3 = read_pc(keyframe3_path)
    # keyframe9 = read_pc(keyframe9_path)
    #
    # keyframe2_xyz = keyframe_poses[2, :]
    # keyframe0_xyz = keyframe_poses[0, :]
    # keyframe1_xyz = keyframe_poses[1, :]
    # keyframe3_xyz = keyframe_poses[3, :]
    # keyframe9_xyz = keyframe_poses[9, :]
    #
    # keyframe2_rot = keyframe_rot[2, :]
    # keyframe0_rot = keyframe_rot[0, :]
    # keyframe1_rot = keyframe_rot[1, :]
    # keyframe3_rot = keyframe_rot[3, :]
    # keyframe9_rot = keyframe_rot[9, :]
    #
    # t10 = keyframe1_rot
    # t20 = keyframe2_rot
    #
    # # keyframe10 = (keyframe1 - keyframe1_xyz) @ np.linalg.inv(keyframe1_rot)
    # # keyframe20 = (keyframe2 - keyframe2_xyz) @ np.linalg.inv(keyframe2_rot)
    # # keyframe30 = (keyframe3 - keyframe3_xyz) @ np.linalg.inv(keyframe3_rot)
    # # keyframe90 = (keyframe9 - keyframe9_xyz) @ np.linalg.inv(keyframe9_rot)
    #
    # keyframe10 = keyframe1 @ np.linalg.inv(keyframe1_rot) + keyframe1_xyz
    # keyframe20 = keyframe2 @ np.linalg.inv(keyframe2_rot) + keyframe2_xyz
    # keyframe30 = keyframe3 @ np.linalg.inv(keyframe3_rot)+ keyframe3_xyz
    # keyframe90 = keyframe9 @ np.linalg.inv(keyframe9_rot) + keyframe9_xyz
    #
    # keyframe01239 = np.concatenate((keyframe0, keyframe10, keyframe20, keyframe30, keyframe90), axis=0)
    # parameters_path = '/home/vectr/PycharmProjects/lidar_learning/configs/parameters.yml'
    # parameters = yaml.safe_load(open(parameters_path))
    # image = compute_range_image(keyframe01239, parameters['lidar'], 1024, 128)
    # plt.imshow(image)
    # plt.show()

    seq = 'e4_1'
    # pcd_files = f'/media/vectr/vectr3/Dataset/loop_closure_detection/keyframes/{seq}/pcd_files'
    # png_files = f'/media/vectr/vectr3/Dataset/loop_closure_detection/keyframes/{seq}/png_files/512'
    pcd_files = f'/media/vectr/vectr3/Dataset/loop_closure_detection/pcd_files/{seq}'
    png_files = f'/media/vectr/vectr3/Dataset/loop_closure_detection/png_files/{seq}/512'

    # parameters_path = '/home/vectr/ws/src/keyframe_detector/src/keyframe_detector/configs/parameters.yml'
    parameters_path = '/home/vectr/PycharmProjects/lidar_learning/configs/parameters.yml'
    paths = [os.path.join(root, file) for root, dirs, files in os.walk(pcd_files) for file in files]
    paths.sort()
    parameters = yaml.safe_load(open(parameters_path))

    import matplotlib.pyplot as plt
    # image plots
    for i, pc_path in tqdm.tqdm(enumerate(paths)):
        pc = o3d.io.read_point_cloud(pc_path)
        pc = np.asarray(pc.points)
        image = compute_range_image(pc, parameters['lidar'], 512, 32)

        # normalize the image
        normalize = False
        if normalize:
            image = image / np.max(image)

        # save the projection as an image
        filename = os.path.join(png_files, f'{str(i).zfill(6)}.png')
        cv2.imwrite(filename, image)

        plt.figure(1)
        plt.clf()
        plt.imshow(image)
        plt.title(f'image: {i}')
        plt.pause(0.01)
    #
    # # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # # model = featureExtracter(height=32, width=900, channels=1, use_transformer=True).to(device)
    # # checkpoint = torch.load('/media/vectr/vectr3/Dataset/overlap_transformer/weights/weights_new/best.pth.tar')
    # # model.load_state_dict(checkpoint['state_dict'])
    # # model.eval()
    # #
    # # for path in tqdm.tqdm(paths[:500]):
    # #     pc = o3d.io.read_point_cloud(path)
    # #     pc = np.asarray(pc.points)
    # #     pc = torch.from_numpy(pc).to(device)
    # #     image = compute_range_image_tensor(pc, parameters['lidar'], 900, 32, device)
    # #     descriptor = compute_descriptor(model, image, device)
