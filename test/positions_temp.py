import os
import torch
import tqdm
import faiss
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from modules.ot_copy.modules.overlap_transformer_haomo import featureExtracter
from modules.overlap_transformer import OverlapTransformer32
from tools.fileloader import load_xyz_rot, read_image, load_files


def calc_T(path):
    xyz, rot = load_xyz_rot(path)
    T = np.zeros((4, 4))
    T[:3, :3] = rot[-1]
    T[:3, 3] = xyz[-1]
    T[3, 3] = 1.0
    return T


def alignment_ground_truth(xyz, T):
    xyz = np.concatenate((xyz, np.ones((len(xyz), 1))), axis=1)
    traj = xyz @ T.T
    traj = traj[:, :3]
    return traj


def alignment_estimated_cluster_2_point(clusters, keyframes):
    if len(clusters) != len(keyframes):
        raise 'Number of clusters and selected keyframes do not match!!!'

    centroids = np.zeros((len(clusters), 3))
    keyframes = np.array(keyframes)
    for i in range(len(clusters)):
        cluster = clusters[i]
        centroid = np.mean(cluster, axis=0, keepdims=True)
        centroids[i, :] = centroid

    R, t = least_squares_points_alignment(centroids, keyframes)
    return R, t


def alignment_estimated_point_2_point(points, keyframes, weights=None):
    if len(points) != len(keyframes):
        raise 'Number of points and selected keyframes do not match!!!'
    R, t = least_squares_points_alignment(points, keyframes, weights)
    return R, t


def calc_descriptors(images, model):
    n_keyframes = len(images)
    descriptors = np.zeros((n_keyframes, 256))

    with torch.no_grad():
        for i in tqdm.tqdm(range(n_keyframes)):
            curr_batch = read_image(images[i])
            curr_batch = torch.cat((curr_batch, curr_batch), dim=0)

            model.eval()
            curr_descriptor = model(curr_batch)
            descriptors[i, :] = curr_descriptor[0, :].cpu().detach().numpy()

    descriptors = descriptors.astype('float32')
    return descriptors


def calc_top_n_descriptors(reference_keyframes_descriptors, test_descriptors, top_n=2):
    dim_descriptors = 256
    index_descriptors = faiss.IndexFlatL2(dim_descriptors)
    index_descriptors.add(reference_keyframes_descriptors)

    # search the top n keyframes based on the distance between the descriptors
    _, top_n_choices = index_descriptors.search(test_descriptors, top_n)
    return top_n_choices


def calc_top_n_distances(reference_keyframes_poses, test_frames_poses, top_n=1):
    dim_poses = 3
    index_poses = faiss.IndexFlatL2(dim_poses)
    index_poses.add(reference_keyframes_poses)

    # search the top n keyframes based on the distance in R3
    _, top_n_choices = index_poses.search(test_frames_poses, top_n)
    return top_n_choices


def calc_top_n_confidence_score(descriptors, descriptors_kf, top_n_choices, use_min=True):
    num_frames = descriptors.shape[0]
    confidence_scores = np.zeros(num_frames)
    for idx in range(len(descriptors)):
        curr_descriptor = descriptors[idx, :]
        top_n_choice = top_n_choices[idx]
        top_n_descriptors = descriptors_kf[top_n_choice, :]
        if use_min:
            top_n_best_dist = np.min(np.linalg.norm(curr_descriptor - top_n_descriptors, axis=1))
        else:
            top_n_best_dist = np.mean(np.linalg.norm(curr_descriptor - top_n_descriptors, axis=1))
        confidence_scores[idx] = top_n_best_dist

    confidence_scores /= np.max(confidence_scores)
    confidence_scores = np.ones_like(confidence_scores) - confidence_scores

    return confidence_scores


def least_squares_points_alignment(src_points, dst_points, weights=None):
    dim = src_points.shape[1]
    num_src_points = src_points.shape[0]
    num_dst_points = dst_points.shape[0]
    if num_src_points != num_dst_points:
        raise ValueError('Number of points do not match!')

    # input points are n * d (row based), switch to d * n (column based)
    src_points = src_points.T
    dst_points = dst_points.T
    if weights is None:
        weights = np.ones((1, num_src_points))

    p_bar = np.sum(weights * src_points, axis=1, keepdims=True) / np.sum(weights)
    q_bar = np.sum(weights * dst_points, axis=1, keepdims=True) / np.sum(weights)

    X = src_points - p_bar
    Y = dst_points - q_bar
    W = np.diagflat(weights)

    S = X @ W @ Y.T
    U, _, Vh = np.linalg.svd(S)
    U_T = U.T
    V = Vh.T

    M = np.diagflat(np.ones(dim))
    M[-1, -1] = np.linalg.det(V @ U_T)

    R = V @ M @ U_T
    t = q_bar - R @ p_bar

    # return R, t are used for row based points (n * d)
    return R.T, t.T


def test_handler(reference_poses_path, reference_keyframes_poses_path, test_frames_poses_path, test_keyframes_poses_path, translation_path,
                 reference_keyframes_img_folder_path, test_frames_img_folder_path, weights_path,
                 top_n_descriptors=2, top_n_distances=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = featureExtracter(height=32, width=512, channels=1, use_transformer=True).to(device)

    with torch.no_grad():
        # load model
        print(f'Loading model from {weights_path}.')
        checkpoint = torch.load(weights_path)
        model.load_state_dict(checkpoint['state_dict'])

        # load reference frames, reference keyframes, test frames positions
        translation_matrix = calc_T(translation_path)
        reference_frames_xyz, _ = load_xyz_rot(reference_poses_path)
        reference_keyframes_xyz, _ = load_xyz_rot(reference_keyframes_poses_path)
        test_frames_xyz, _ = load_xyz_rot(test_frames_poses_path)
        test_keyframes_xyz, _ = load_xyz_rot(test_keyframes_poses_path)
        test_frames_xyz_aligned = alignment_ground_truth(test_frames_xyz, translation_matrix)    # test frames positions in reference coord
        test_keyframes_xyz_aligned = alignment_ground_truth(test_keyframes_xyz, translation_matrix)

        # load reference keyframes images and test frames images
        reference_keyframes_img_paths = load_files(reference_keyframes_img_folder_path)
        test_frames_img_paths = load_files(test_frames_img_folder_path)

        reference_keyframes_descriptors = calc_descriptors(reference_keyframes_img_paths, model)
        test_frames_descriptors = calc_descriptors(test_frames_img_paths, model)

        # search the top n keyframes in both feature space and euclidean space (indices)
        choices_descriptors = calc_top_n_descriptors(reference_keyframes_descriptors, test_frames_descriptors, top_n=top_n_descriptors)
        choices_distances = calc_top_n_distances(reference_keyframes_xyz, test_frames_xyz_aligned, top_n=top_n_distances)

        # calculate confidence scores
        confidence_scores = calc_top_n_confidence_score(test_frames_descriptors, reference_keyframes_descriptors,
                                                        choices_descriptors, use_min=True)

        pos_pred = []
        neg_pred = []
        predictions_mask = []
        num_correct_pred = 0

        for i in range(len(choices_descriptors)):
            top_n_choices_descriptors = choices_descriptors[i, :]
            top_n_choices_distances = choices_distances[i, :]
            for j in range(len(top_n_choices_distances)):
                if top_n_choices_distances[j] in top_n_choices_descriptors:
                    pos_pred.append(i)
                    predictions_mask.append(1)
                    num_correct_pred += 1
                    break

                if j == len(top_n_choices_distances) - 1:
                    neg_pred.append(i)
                    predictions_mask.append(0)

        precision = num_correct_pred / len(choices_descriptors)
        print(f'Prediction precision: {precision:.3f}.')

        # alignment estimation
        # align second trajectory based on the prediction
        # first piece = pos[40:100] -> keyframe[-3]
        # second piece = pos[111:195] -> keyframe[-1]
        # third piece = pos[200:295] -> keyframe[-3]
        # special piece (close failed case) = neg[600:634] -> keyframe[-2]
        pos_xyz_unaligned = test_frames_xyz[pos_pred, :]
        neg_xyz_unaligned = test_frames_xyz[neg_pred, :]
        pos_first_piece = pos_xyz_unaligned[40:100, :]
        pos_second_piece = pos_xyz_unaligned[111:195, :]
        pos_third_piece = pos_xyz_unaligned[200:295, :]
        neg_forth_piece = neg_xyz_unaligned[600:634, :]

        # align by clusters and keyframes
        clusters = [pos_first_piece, pos_second_piece, pos_third_piece,
                    neg_forth_piece]
        clusters_keyframes = [reference_keyframes_xyz[-3], reference_keyframes_xyz[-1], reference_keyframes_xyz[-3],
                              reference_keyframes_xyz[-2]]

        # align by confidence scores (point to point alignment)
        confidence_threshold = 0.4
        confidence_mask = confidence_scores > confidence_threshold
        confidence_points = test_frames_xyz[confidence_mask, :]
        confidence_keyframes = reference_keyframes_xyz[choices_descriptors[confidence_mask, 0]]     # 0: only use top 1

        # R, t = alignment_estimated_cluster_2_point(clusters, clusters_keyframes)
        # test_frames_xyz_estimated = test_frames_xyz @ R + t

        R, t = alignment_estimated_point_2_point(confidence_points, confidence_keyframes, weights=confidence_scores[confidence_mask])
        test_frames_xyz_estimated = test_frames_xyz @ R + t


        # plot
        # plot_method: aligned, unaligned, estimated
        # plot_skip: avoid dense connection lines
        pos_xyz = test_frames_xyz_aligned[pos_pred, :]
        neg_xyz = test_frames_xyz_aligned[neg_pred, :]
        plot_method = 'estimated'
        plot_skip = 20

        colors = ['blue', 'orange', 'gold', 'red', 'purple', 'brown', 'pink', 'violet', 'cyan']
        test_colors = [colors[i] for i in choices_descriptors[:, 0]]
        test_lines = []
        test_lines_color = []

        if plot_method == 'aligned':
            test_frames_plot = test_frames_xyz_aligned
            title = 'Prediction (Aligned)'
        elif plot_method == 'unaligned':
            test_frames_plot = test_frames_xyz
            title = 'Prediction (Unaligned)'
        elif plot_method == 'estimated':
            test_frames_plot = test_frames_xyz_estimated
            title = 'Prediction (Estimated)'
        else:
            raise ValueError(f'Method {plot_method} not recognized!')

        # add connection lines
        for i in range(0, len(choices_descriptors), plot_skip):
            start_point = test_frames_plot[i, :]
            end_point = reference_keyframes_xyz[choices_descriptors[i, 0], :]
            test_lines.append([start_point[:2], end_point[:2]])
            test_lines_color.append(test_colors[i])
        test_lines_collections = matplotlib.collections.LineCollection(test_lines, colors=test_lines_color, linewidths=1,
                                                                       linestyles='dashdot')

        # ground truth alignment prediction
        fig, [ax1, ax2] = plt.subplots(1, 2)
        ax1.scatter(reference_frames_xyz[:, 0], reference_frames_xyz[:, 1], c='violet', s=10, label='Reference Trajectory')
        ax1.scatter(test_frames_xyz_aligned[:, 0], test_frames_xyz_aligned[:, 1], c='grey', s=10, label='Test Ground Truth')
        ax1.scatter(test_frames_plot[:, 0], test_frames_plot[:, 1], c='blue', s=10, label='Test Prediction')
        ax1.scatter(reference_keyframes_xyz[:, 0], reference_keyframes_xyz[:, 1], c=colors, s=20)
        # ax.scatter(test_frames_xyz_aligned[:, 0], test_frames_xyz_aligned[:, 1], c=test_colors)
        # ax1.scatter(pos_xyz[:, 0], pos_xyz[:, 1], c='green', s=0.5)
        # ax1.scatter(neg_xyz[:, 0], neg_xyz[:, 1], c='red', s=0.5)
        # ax1.add_collection(test_lines_collections)
        ax1.set_xlabel('x (m)', fontsize=15)
        ax1.set_ylabel('y (m)', fontsize=15)
        ax1.set_title(title, fontsize=20)
        ax1.legend(fontsize=15)

        # plot for confidence
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)
        mapper = matplotlib.cm.ScalarMappable(norm=norm)
        mapper.set_array(confidence_scores)
        colors = np.array([mapper.to_rgba(c) for c in confidence_scores])

        ax2.scatter(reference_frames_xyz[:, 0], reference_frames_xyz[:, 1], c='violet', s=10, label='Reference Trajectory')
        ax2.scatter(test_frames_plot[:, 0], test_frames_plot[:, 1], c=colors, s=10)
        ax2.set_xlabel('x [m]', fontsize=15)
        ax2.set_ylabel('y [m]', fontsize=15)
        ax2.set_title('Confidence', fontsize=20)
        cbar = plt.colorbar(mapper)
        # cbar.set_label('Confidence', rotation=270, weight='bold')

        # plt.legend(fontsize=15)
        plt.show()

        return pos_pred, neg_pred


def test_handler_2(reference_poses_path, reference_keyframes_poses_path, test_frames_poses_path, test_frames_gt_poses,
                   translation_path, reference_keyframes_img_folder_path, test_frames_img_folder_path, weights_path,
                   top_n_descriptors=2, top_n_distances=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OverlapTransformer32(height=32, width=512, channels=1, use_transformer=True).to(device)

    with torch.no_grad():
        # load model
        print(f'Loading model from {weights_path}.')
        checkpoint = torch.load(weights_path)
        model.load_state_dict(checkpoint['state_dict'])

        # load reference frames, reference keyframes, test frames positions
        translation_matrix = calc_T(translation_path)
        reference_frames_xyz, _ = load_xyz_rot(reference_poses_path)
        reference_keyframes_xyz, _ = load_xyz_rot(reference_keyframes_poses_path)
        test_frames_xyz, _ = load_xyz_rot(test_frames_poses_path)
        test_frames_xyz_aligned = alignment_ground_truth(test_frames_xyz, translation_matrix)    # test frames positions in reference coord
        test_frames_xyz_gt, _ = load_xyz_rot(test_frames_gt_poses)

        # load reference keyframes images and test frames images
        reference_keyframes_img_paths = load_files(reference_keyframes_img_folder_path)
        test_frames_img_paths = load_files(test_frames_img_folder_path)

        reference_keyframes_descriptors = calc_descriptors(reference_keyframes_img_paths, model)
        test_frames_descriptors = calc_descriptors(test_frames_img_paths, model)

        # search the top n keyframes in both feature space and euclidean space (indices)
        choices_descriptors = calc_top_n_descriptors(reference_keyframes_descriptors, test_frames_descriptors, top_n=top_n_descriptors)
        choices_distances = calc_top_n_distances(reference_keyframes_xyz, test_frames_xyz_aligned, top_n=top_n_distances)

        # calculate confidence scores
        confidence_scores = calc_top_n_confidence_score(test_frames_descriptors, reference_keyframes_descriptors,
                                                        choices_descriptors, use_min=True)

        pos_pred = []
        neg_pred = []
        predictions_mask = []
        num_correct_pred = 0

        for i in range(len(choices_descriptors)):
            top_n_choices_descriptors = choices_descriptors[i, :]
            top_n_choices_distances = choices_distances[i, :]
            for j in range(len(top_n_choices_distances)):
                if top_n_choices_distances[j] in top_n_choices_descriptors:
                    pos_pred.append(i)
                    predictions_mask.append(1)
                    num_correct_pred += 1
                    break

                if j == len(top_n_choices_distances) - 1:
                    neg_pred.append(i)
                    predictions_mask.append(0)

        precision = num_correct_pred / len(choices_descriptors)
        print(f'Prediction precision: {precision:.3f}.')

        # alignment estimation
        # align second trajectory based on the prediction
        # first piece = pos[40:100] -> keyframe[-3]
        # second piece = pos[111:195] -> keyframe[-1]
        # third piece = pos[200:295] -> keyframe[-3]
        # special piece (close failed case) = neg[600:634] -> keyframe[-2]
        pos_xyz_unaligned = test_frames_xyz[pos_pred, :]
        neg_xyz_unaligned = test_frames_xyz[neg_pred, :]
        pos_first_piece = pos_xyz_unaligned[40:100, :]
        pos_second_piece = pos_xyz_unaligned[111:195, :]
        pos_third_piece = pos_xyz_unaligned[200:295, :]
        neg_forth_piece = neg_xyz_unaligned[600:634, :]

        # align by clusters and keyframes
        clusters = [pos_first_piece, pos_second_piece, pos_third_piece,
                    neg_forth_piece]
        clusters_keyframes = [reference_keyframes_xyz[-3], reference_keyframes_xyz[-1], reference_keyframes_xyz[-3],
                              reference_keyframes_xyz[-2]]

        # align by confidence scores (point to point alignment)
        confidence_threshold = 0.4
        confidence_mask = confidence_scores > confidence_threshold
        confidence_points = test_frames_xyz[confidence_mask, :]
        confidence_keyframes = reference_keyframes_xyz[choices_descriptors[confidence_mask, 0]]     # 0: only use top 1

        # R, t = alignment_estimated_cluster_2_point(clusters, clusters_keyframes)
        # test_frames_xyz_estimated = test_frames_xyz @ R + t

        R, t = alignment_estimated_point_2_point(confidence_points, confidence_keyframes, weights=confidence_scores[confidence_mask])
        test_frames_xyz_estimated = test_frames_xyz @ R + t


        # plot
        # plot_method: aligned, unaligned, estimated
        # plot_skip: avoid dense connection lines
        pos_xyz = test_frames_xyz_aligned[pos_pred, :]
        neg_xyz = test_frames_xyz_aligned[neg_pred, :]
        plot_method = 'estimated'
        plot_skip = 20

        # colors = ['blue', 'orange', 'gold', 'red', 'purple', 'brown', 'pink', 'violet', 'cyan']
        # test_colors = [colors[i] for i in choices_descriptors[:, 0]]
        test_lines = []
        test_lines_color = []

        if plot_method == 'aligned':
            test_frames_plot = test_frames_xyz_aligned
            title = 'Prediction (Aligned)'
        elif plot_method == 'unaligned':
            test_frames_plot = test_frames_xyz
            title = 'Prediction (Unaligned)'
        elif plot_method == 'estimated':
            test_frames_plot = test_frames_xyz_estimated
            title = 'Prediction (Estimated)'
        else:
            raise ValueError(f'Method {plot_method} not recognized!')

        # add connection lines
        for i in range(0, len(choices_descriptors), plot_skip):
            start_point = test_frames_plot[i, :]
            end_point = reference_keyframes_xyz[choices_descriptors[i, 0], :]
            test_lines.append([start_point[:2], end_point[:2]])
            test_lines_color.append('pink')
        test_lines_collections = matplotlib.collections.LineCollection(test_lines, colors=test_lines_color, linewidths=1,
                                                                       linestyles='dashdot')

        # ground truth alignment prediction
        fig, [ax1, ax2] = plt.subplots(1, 2)
        ax1.scatter(reference_frames_xyz[:, 0], reference_frames_xyz[:, 1], c='pink', s=10, label='Reference Trajectory')
        ax1.scatter(test_frames_xyz[:, 0], test_frames_xyz[:, 1], c='grey', s=10, label='Test Trajectory')
        # ax1.scatter(test_frames_xyz_gt[:, 0], test_frames_xyz_gt[:, 1], c='violet', s=10, label='Test Ground Truth')
        ax1.scatter(test_frames_plot[:, 0], test_frames_plot[:, 1], c='blue', s=10, label='Test Prediction')
        # ax1.scatter(reference_keyframes_xyz[:, 0], reference_keyframes_xyz[:, 1], c='pink', s=20)
        # ax.scatter(test_frames_xyz_aligned[:, 0], test_frames_xyz_aligned[:, 1], c=test_colors)
        # ax1.scatter(pos_xyz[:, 0], pos_xyz[:, 1], c='green', s=0.5)
        # ax1.scatter(neg_xyz[:, 0], neg_xyz[:, 1], c='red', s=0.5)
        # ax1.add_collection(test_lines_collections)
        ax1.set_xlabel('x (m)', fontsize=15)
        ax1.set_ylabel('y (m)', fontsize=15)
        ax1.set_title(title, fontsize=20)
        ax1.legend(fontsize=15)

        # plot for confidence
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)
        mapper = matplotlib.cm.ScalarMappable(norm=norm)
        mapper.set_array(confidence_scores)
        colors = np.array([mapper.to_rgba(c) for c in confidence_scores])

        # ax2.scatter(reference_frames_xyz[:, 0], reference_frames_xyz[:, 1], c='violet', s=10, label='Reference Trajectory')
        ax2.scatter(test_frames_xyz_gt[:, 0], test_frames_xyz_gt[:, 1], c='violet', s=10, label='Test Ground Truth')
        ax2.scatter(test_frames_plot[:, 0], test_frames_plot[:, 1], c=colors, s=10)
        ax2.set_xlabel('x [m]', fontsize=15)
        ax2.set_ylabel('y [m]', fontsize=15)
        ax2.set_title('Confidence', fontsize=20)
        ax2.legend(fontsize=15)
        cbar = plt.colorbar(mapper)
        # cbar.set_label('Confidence', rotation=270, weight='bold')

        # plt.legend(fontsize=15)
        plt.show()

        return pos_pred, neg_pred


if __name__ == '__main__':
    # traj1_poses = '/media/vectr/vectr3/Dataset/loop_closure_detection/poses/e4_1/poses.txt'
    # traj2_poses = '/media/vectr/vectr3/Dataset/loop_closure_detection/poses/e4_3/poses.txt'
    # traj1_kf_poses = '/media/vectr/vectr3/Dataset/loop_closure_detection/poses/e4_1/poses_kf.txt'
    # traj2_kf_poses = '/media/vectr/vectr3/Dataset/loop_closure_detection/poses/e4_3/poses_kf.txt'
    # translation = '/media/vectr/vectr3/Dataset/loop_closure_detection/poses/e4_2/poses.txt'
    #
    # traj1_img = '/media/vectr/vectr3/Dataset/loop_closure_detection/png_files/e4_1/512'
    # traj2_img = '/media/vectr/vectr3/Dataset/loop_closure_detection/png_files/e4_3/512'
    # traj1_kf_img = '/media/vectr/vectr3/Dataset/loop_closure_detection/keyframes/e4_1/png_files/512'
    # traj2_kf_img = '/media/vectr/vectr3/Dataset/loop_closure_detection/keyframes/e4_3/png_files/512'
    #
    # weights = '/media/vectr/vectr3/Dataset/overlap_transformer/weights/weights_512_schedule/last.pth.tar'
    #
    # choices_descriptors, choices_distances = test_handler(traj1_poses, traj1_kf_poses, traj2_poses, traj2_kf_poses, translation,
    #              traj1_kf_img, traj2_img, weights, top_n_descriptors=1, top_n_distances=1)

    traj1_poses = '/media/vectr/vectr3/Dataset/arl/test_alignment/out-and-back-3/poses/poses_1.txt'
    traj2_poses = '/media/vectr/vectr3/Dataset/arl/test_alignment/out-and-back-3/poses/poses_2.txt'
    traj2_gt_poses = '/media/vectr/vectr3/Dataset/arl/test_alignment/out-and-back-3/poses/poses_2_gt.txt'
    traj1_kf_poses = '/media/vectr/vectr3/Dataset/arl/test_alignment/out-and-back-3/poses/poses_1_kf.txt'
    translation = '/media/vectr/vectr3/Dataset/arl/test_alignment/out-and-back-3/poses/poses_1.txt'

    traj1_img = '/media/vectr/vectr3/Dataset/arl/test_alignment/out-and-back-3/png_files/512_1'
    traj2_img = '/media/vectr/vectr3/Dataset/arl/test_alignment/out-and-back-3/png_files/512_2'
    traj1_kf_img = '/media/vectr/vectr3/Dataset/arl/test_alignment/out-and-back-3/png_files/512_1_kf'

    weights_path = '/media/vectr/vectr3/Dataset/overlap_transformer/weights/weights_06_26'
    weights = os.path.join(weights_path, 'best.pth.tar')

    test_handler_2(traj1_poses, traj1_kf_poses, traj2_poses, traj2_gt_poses, translation, traj1_kf_img, traj2_img, weights,
                   top_n_descriptors=5, top_n_distances=1)
