import os
import tqdm
import yaml
import cv2

from tools.fileloader import load_files, load_submaps, load_xyz_rot, read_pc
from data.kitti.dataset.projection import RangeProjection


def submaps_projection(submaps, pcd_files_path, poses_path, dst_path):
    pcd_files = load_files(pcd_files_path)
    xyz, rot = load_xyz_rot(poses_path)
    projector = RangeProjection(proj_w=512, proj_h=64)

    for i in tqdm.tqdm(range(submaps.shape[0])):
        # saving directory
        saving_path = os.path.join(dst_path, f'{str(i).zfill(6)}')
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)

        for j in range(submaps.shape[1]):
            # transform
            pc_index = submaps[i, j]
            pc_world = read_pc(pcd_files[pc_index])
            pc_local = pc_world @ rot[i] - xyz[i] @ rot[i]
            _, img_local, _, _ = projector.doProjection(pc_local)
            cv2.imwrite(os.path.join(saving_path, f'{str(j)}.png'), img_local)


if __name__ == '__main__':
    data_root = '/Volumes/T7/Datasets/public_datasets/kitti/dataset/sequences'
    params_path = '/Users/yanlong/PycharmProjects/lidar_learning/data/kitti/dataset/config_kitti.yml'
    params = yaml.safe_load(open(params_path))
    sequences = ['14', '15', '16', '17', '18', '19', '20', '21']

    for sequence in sequences:
        # src files
        pcd_files_path = os.path.join(data_root, sequence, 'pcd_files')
        poses_path = os.path.join(data_root, sequence, 'poses/poses.txt')
        submaps_paths = [os.path.join(data_root, sequence, "submaps/overlap_heuristic"),
                         os.path.join(data_root, sequence, "submaps/overlap"),
                         os.path.join(data_root, sequence, "submaps/euclidean_heuristic"),
                         os.path.join(data_root, sequence, "submaps/euclidean")
                         ]

        for submaps_path in submaps_paths:
            # dst directory
            anchor_dst_path = os.path.join(submaps_path, 'anchor/512/orig')
            pos_neg_dst_path = os.path.join(submaps_path, 'pos_neg/512/orig')
            kf_dst_path = os.path.join(submaps_path, 'kf/512/orig')

            if not os.path.exists(anchor_dst_path):
                os.makedirs(anchor_dst_path)
            if not os.path.exists(pos_neg_dst_path):
                os.makedirs(pos_neg_dst_path)
            if not os.path.exists(kf_dst_path):
                os.makedirs(kf_dst_path)

            # load subamps matrices
            anchor, pos_neg, kf = load_submaps(submaps_path)

            submaps_projection(anchor, pcd_files_path, poses_path, anchor_dst_path)
            submaps_projection(pos_neg, pcd_files_path, poses_path, pos_neg_dst_path)
            submaps_projection(kf, pcd_files_path, poses_path, kf_dst_path)
