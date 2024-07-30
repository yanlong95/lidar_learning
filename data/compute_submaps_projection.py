import os
import tqdm
import yaml
import cv2

from tools.fileloader import load_files, load_submaps, load_xyz_rot, read_pc
from projection.range_image import range_projection


def submaps_projection(submaps, pcd_files_path, poses_path, dst_path, params):
    pcd_files = load_files(pcd_files_path)
    xyz, rot = load_xyz_rot(poses_path)

    for i in tqdm.tqdm(range(submaps.shape[0])):
        # saving directory
        saving_path = os.path.join(dst_path, f'{str(i).zfill(6)}')
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)

        for j in range(submaps.shape[1]):
            # transform
            pc_index = submaps[i, j]
            pc_lidar = read_pc(pcd_files[pc_index])
            pc_world = pc_lidar @ rot[pc_index].T + xyz[pc_index]
            pc_local = pc_world @ rot[i] - xyz[i] @ rot[i]
            img_local = range_projection(pc_local, params, 32, 512)
            cv2.imwrite(os.path.join(saving_path, f'{str(j)}.png'), img_local)


if __name__ == '__main__':
    config_path = '/home/vectr/PycharmProjects/lidar_learning/configs/config.yml'
    params_path = '/home/vectr/PycharmProjects/lidar_learning/configs/parameters.yml'

    config = yaml.safe_load(open(config_path))
    # sequences = config['seqs']['all']
    sequences = ["bomb_shelter", "botanical_garden", "bruin_plaza", "court_of_sciences", "dickson_court", "geo_loop"]

    for sequence in sequences:
        # src files
        pcd_files_path = os.path.join(config['data_root']['pcd_files'], sequence)
        # submaps_path = os.path.join(config['data_root']['submaps'], sequence)
        poses_path = os.path.join(config['data_root']['poses'], sequence, 'poses.txt')
        submaps_paths = ["/media/vectr/vectr3/Dataset/overlap_transformer/submaps/overlap"]
        # submaps_paths = ["/media/vectr/vectr3/Dataset/overlap_transformer/submaps/overlap_heuristic",
        #                  "/media/vectr/vectr3/Dataset/overlap_transformer/submaps/overlap",
        #                  "/media/vectr/vectr3/Dataset/overlap_transformer/submaps/euclidean_heuristic",
        #                  "/media/vectr/vectr3/Dataset/overlap_transformer/submaps/euclidean"
        #                  ]

        for submaps_path in submaps_paths:

            submaps_path = os.path.join(submaps_path, sequence)

            # dst directory
            anchor_dst_path = os.path.join(submaps_path, 'anchor')
            # pos_neg_dst_path = os.path.join(submaps_path, 'pos_neg')
            # kf_dst_path = os.path.join(submaps_path, 'kf')

            if not os.path.exists(anchor_dst_path):
                os.makedirs(anchor_dst_path)
            # if not os.path.exists(pos_neg_dst_path):
            #     os.makedirs(pos_neg_dst_path)
            # if not os.path.exists(kf_dst_path):
            #     os.makedirs(kf_dst_path)

            # load subamps matrices
            anchor, pos_neg, kf = load_submaps(submaps_path)

            # transform keyframes point clouds from their local coordinate to current scan local coordinate]
            if sequence in ["botanical_garden", "bruin_plaza", "court_of_sciences", "dickson_court", "royce_hall",
                            "sculpture_garden"]:
                params = {'fov_up': 22.5, 'fov_down': -22.5, 'eps': 1e-8}
            elif sequence in ["bomb_shelter", "geo_loop", "kerckhoff", "luskin"]:
                params = {'fov_up': 45, 'fov_down': -45, 'eps': 1e-8}
            else:
                raise ValueError(f'Invalid sequence name: {sequence}')

            submaps_projection(anchor, pcd_files_path, poses_path, anchor_dst_path, params)
            # submaps_projection(pos_neg, pcd_files_path, poses_path, pos_neg_dst_path, params)
            # submaps_projection(kf, pcd_files_path, poses_path, kf_dst_path, params)
