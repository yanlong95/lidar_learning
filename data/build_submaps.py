"""
File to build submaps for each scan in a dataset. The built subamps are saved as a numpy matrix.
anchor submaps are built using only the previous keyframes.
positive and negative submaps are built using all keyframes.
keyframe submaps are built using only keyframes.
Both anchor and pos_neg submaps are in all frames indices orders.
Keyframe submaps are in keyframes indices order.
"""
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

from compute_submaps import compute_submap_keyframes, compute_keyframes_indices
from tools.fileloader import load_xyz_rot, load_overlaps


if __name__ == '__main__':
    config_path = '/home/vectr/PycharmProjects/lidar_learning/configs/config.yml'
    params_path = '/home/vectr/PycharmProjects/lidar_learning/configs/parameters.yml'

    config = yaml.safe_load(open(config_path))
    params = yaml.safe_load(open(params_path))

    frames_poses_folder = config['data_root']['poses']
    keyframes_poses_folder = config['data_root']['keyframes']
    overlaps_folder = config['data_root']['overlaps']
    submaps_folder = config['data_root']['submaps']

    sequences = config['seqs']['all']

    # if True, the submaps are in keyframes order. Otherwise, the submaps are in all frames order.
    in_keyframes_order = False

    for sequence in sequences:
        frames_poses_path = os.path.join(frames_poses_folder, sequence, 'poses.txt')
        keyframes_poses_path = os.path.join(keyframes_poses_folder, sequence, 'poses', 'poses_kf.txt')
        overlaps_path = os.path.join(overlaps_folder, f'{sequence}.bin')

        # load the poses of the frames
        xyz, _ = load_xyz_rot(frames_poses_path)
        xyz_kf, _ = load_xyz_rot(keyframes_poses_path)
        overlaps = load_overlaps(overlaps_path)

        # compute the keyframes indices
        indices_kf = compute_keyframes_indices(xyz, xyz_kf)     # compute the indices of keyframes (in number of frames)

        # compute the submaps for all frames (in shape (n, top_k))
        submaps_euclidean_in_kf_order = compute_submap_keyframes(xyz, xyz_kf, overlaps, top_k=5, overlap_dist_thresh=75.0,
                                                     is_anchor=False, metric='euclidean')
        submaps_overlap_in_kf_order = compute_submap_keyframes(xyz, xyz_kf, overlaps, top_k=5, overlap_dist_thresh=75.0,
                                                   is_anchor=False, metric='overlap')

        # compute the submaps for anchor (in shape (n, top_k)). Anchor submaps only consider the previous keyframes
        submaps_anchor_euclidean_in_kf_order = compute_submap_keyframes(xyz, xyz_kf, overlaps, top_k=5, overlap_dist_thresh=75.0,
                                                            is_anchor=True, metric='euclidean')
        submaps_anchor_overlap_in_kf_order = compute_submap_keyframes(xyz, xyz_kf, overlaps, top_k=5, overlap_dist_thresh=75.0,
                                                          is_anchor=True, metric='overlap')

        # transform the submaps indices from keyframes indices order to all frames indices order
        if in_keyframes_order:
            submaps_euclidean = submaps_euclidean_in_kf_order
            submaps_overlap = submaps_overlap_in_kf_order
            submaps_anchor_euclidean = submaps_anchor_euclidean_in_kf_order
            submaps_anchor_overlap = submaps_anchor_overlap_in_kf_order
        else:
            submaps_euclidean = np.array([indices_kf[submaps_euclidean_in_kf_order[i, :]] for i in
                                          range(len(submaps_euclidean_in_kf_order))])
            submaps_overlap = np.array([indices_kf[submaps_overlap_in_kf_order[i, :]] for i in
                                        range(len(submaps_overlap_in_kf_order))])

            submaps_anchor_euclidean = np.array([indices_kf[submaps_anchor_euclidean_in_kf_order[i, :]] for i in
                                                 range(len(submaps_anchor_euclidean_in_kf_order))])
            submaps_anchor_overlap = np.array([indices_kf[submaps_anchor_overlap_in_kf_order[i, :]] for i in
                                               range(len(submaps_anchor_overlap_in_kf_order))])

        # compute the submaps for keyframes (in shape (m, 3)). indices are in keyframes list order.
        submaps_kf_euclidean = submaps_euclidean_in_kf_order[indices_kf, :]
        submaps_kf_overlap = submaps_overlap_in_kf_order[indices_kf, :]

        # save the submaps
        submaps_euclidean_saving_path = os.path.join(submaps_folder, 'euclidean', sequence)
        submaps_overlap_saving_path = os.path.join(submaps_folder, 'overlap', sequence)

        if not os.path.exists(submaps_euclidean_saving_path):
            os.makedirs(submaps_euclidean_saving_path)
        if not os.path.exists(submaps_overlap_saving_path):
            os.makedirs(submaps_overlap_saving_path)

        np.save(os.path.join(submaps_euclidean_saving_path, 'anchor.npy'), submaps_anchor_euclidean)
        np.save(os.path.join(submaps_euclidean_saving_path, 'pos_neg.npy'), submaps_euclidean)
        np.save(os.path.join(submaps_euclidean_saving_path, 'kf.npy'), submaps_kf_euclidean)
        np.save(os.path.join(submaps_overlap_saving_path, 'anchor.npy'), submaps_anchor_overlap)
        np.save(os.path.join(submaps_overlap_saving_path, 'pos_neg.npy'), submaps_overlap)
        np.save(os.path.join(submaps_overlap_saving_path, 'kf.npy'), submaps_kf_overlap)

    # # visualize generated submaps
    # submap_path = '/media/vectr/vectr3/Dataset/overlap_transformer/submaps/overlap/botanical_garden/anchor.npy'
    # poses_path = '/media/vectr/vectr3/Dataset/overlap_transformer/poses/botanical_garden/poses.txt'
    # poses_kf_path = '/media/vectr/vectr3/Dataset/overlap_transformer/keyframes/botanical_garden/poses/poses_kf.txt'
    #
    # xyz, _ = load_xyz_rot(poses_path)
    # xyz_kf, _ = load_xyz_rot(poses_kf_path)
    # submaps = np.load(submap_path)
    #
    # for i in range(xyz.shape[0]):
    #     xyz_curr = xyz[i, :]
    #     xyz_curr_kf = xyz[submaps[i, :], :]
    #
    #     plt.clf()
    #     plt.scatter(xyz[:, 0], xyz[:, 1])
    #     plt.scatter(xyz_kf[:, 0], xyz_kf[:, 1], c='gold')
    #     plt.scatter(xyz_curr_kf[:, 0], xyz_curr_kf[:, 1], c='red')
    #     plt.scatter(xyz_curr[0], xyz_curr[1], c='violet')
    #     plt.show()
    #     plt.pause(0.01)
