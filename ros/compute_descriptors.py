import os
import yaml
import cv2
import tqdm
import torch
import numpy as np

from tools.fileloader import read_pc, load_files, read_image
from tools.projection import RangeProjection
from modules.overlap_transformer import OverlapTransformer32


def compute_descriptors(img_paths, model, descriptor_size=256):
    num_scans = len(img_paths)
    descriptors = np.zeros((num_scans, descriptor_size), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for i in tqdm.tqdm(range(num_scans)):
            curr_batch = read_image(img_paths[i])
            curr_batch = torch.cat((curr_batch, curr_batch), dim=0)
            curr_descriptors = model(curr_batch)
            descriptors[i, :] = curr_descriptors[0, :].cpu().detach().numpy()

    descriptors = descriptors.astype(np.float32)
    return descriptors


if __name__ == '__main__':
    pcd_folder_path = '/media/vectr/vectr7/newer_college/parkland_mount/pcd_files/mount3'
    png_folder_path = '/media/vectr/vectr7/newer_college/parkland_mount/png_files/mount3'
    params_path = '/home/vectr/PycharmProjects/lidar_learning/configs/datasets.yml'
    dataset = 'newer_college_2020'

    # load parameters and pcd files
    params = yaml.safe_load(open(params_path))
    pcd_files = load_files(pcd_folder_path)

    # define projector for range image projection
    projector = RangeProjection(fov_up=params[dataset]['fov_up'], fov_down=params[dataset]['fov_down'], proj_w=512, proj_h=32)

    # project point clouds to range images
    for i in tqdm.tqdm(range(len(pcd_files))):
        pc = read_pc(pcd_files[i])
        _, img, _, _ = projector.doProjection(pc)
        cv2.imwrite(os.path.join(png_folder_path, f'{str(i).zfill(6)}.png'), img)

    # load model
    weights_path = '/media/vectr/vectr3/Dataset/overlap_transformer/weights/weights_07_01/best.pth.tar'
    checkpoint = torch.load(weights_path)
    model = OverlapTransformer32().to('cuda')
    model.load_state_dict(checkpoint['state_dict'])

    img_paths = load_files(png_folder_path)
    descriptors = compute_descriptors(img_paths, model)

    # save descriptors
    save_path = '/home/vectr/Desktop/temp_desc/parkland_mount3.txt'
    np.savetxt(save_path, descriptors)
