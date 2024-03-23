#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: a simple example to normalize the overlap data, one could do the same to yaw

import numpy as np
import tqdm


def normalize_data_per_frame(ground_truth_mapping, max_pos_thresh=21, min_neg_thresh=7, neg_pos_threshold=3):
    """Normalize the training data according to the overlap value.
       Args:
         ground_truth_mapping: the raw ground truth mapping array
       Returns:
         dist_norm_data: normalized ground truth mapping array
    """
    gt_map = ground_truth_mapping
    bin_0_9 = gt_map[np.where(gt_map[:, 2] < 0.1)]
    bin_10_19 = gt_map[(gt_map[:, 2] < 0.2) & (gt_map[:, 2] >= 0.1)]
    bin_20_29 = gt_map[(gt_map[:, 2] < 0.3) & (gt_map[:, 2] >= 0.2)]
    bin_30_39 = gt_map[(gt_map[:, 2] < 0.4) & (gt_map[:, 2] >= 0.3)]
    bin_40_49 = gt_map[(gt_map[:, 2] < 0.5) & (gt_map[:, 2] >= 0.4)]
    bin_50_59 = gt_map[(gt_map[:, 2] < 0.6) & (gt_map[:, 2] >= 0.5)]
    bin_60_69 = gt_map[(gt_map[:, 2] < 0.7) & (gt_map[:, 2] >= 0.6)]
    bin_70_79 = gt_map[(gt_map[:, 2] < 0.8) & (gt_map[:, 2] >= 0.7)]
    bin_80_89 = gt_map[(gt_map[:, 2] < 0.9) & (gt_map[:, 2] >= 0.8)]
    bin_90_100 = gt_map[(gt_map[:, 2] <= 1) & (gt_map[:, 2] >= 0.9)]

    # keep different bins the same amount of samples
    bins = [bin_0_9, bin_10_19, bin_20_29, bin_30_39, bin_40_49, bin_50_59, bin_60_69, bin_70_79, bin_80_89, bin_90_100]

    # neg_pos_threshold = 3   # e.g. 0.0 ~ 0.3 are negative, 0.3 ~ 1.0 are positive
    dist_norm_data_per_frame_neg = np.empty((0, 4))
    dist_norm_data_per_frame_pos = np.empty((0, 4))

    # add negative samples
    neg_bins_counters = [len(bin_i) for bin_i in bins[:neg_pos_threshold]]

    num_neg_bins_left = neg_pos_threshold
    num_neg_sample_left = neg_pos_threshold * min_neg_thresh
    num_neg_samples = np.zeros(neg_pos_threshold, dtype=int)

    while num_neg_bins_left > 0:
        min_neg_bin_idx = np.argmin(neg_bins_counters)
        min_neg_bin_counter = neg_bins_counters[min_neg_bin_idx]

        num_samples_curr_bin = min(min_neg_bin_counter, num_neg_sample_left // num_neg_bins_left)
        num_neg_samples[min_neg_bin_idx] = num_samples_curr_bin

        num_neg_bins_left -= 1
        num_neg_sample_left -= num_samples_curr_bin
        neg_bins_counters[min_neg_bin_idx] = 2 ** 63 - 1

    for i in range(neg_pos_threshold):
        bin_i = bins[i][np.random.choice(len(bins[i]), num_neg_samples[i], replace=False)]
        dist_norm_data_per_frame_neg = np.vstack((dist_norm_data_per_frame_neg, bin_i))

    # add positive sample
    pos_bins_counters = [len(bin_i) for bin_i in bins[neg_pos_threshold:]]

    if np.sum(pos_bins_counters) < max_pos_thresh:
        non_empty_idx = np.nonzero(pos_bins_counters)[0] + neg_pos_threshold
        for idx in non_empty_idx:
            dist_norm_data_per_frame_pos = np.vstack((dist_norm_data_per_frame_pos, bins[idx]))
    else:
        num_pos_bins_left = len(pos_bins_counters)
        num_pos_sample_left = max_pos_thresh
        num_pos_samples = np.zeros(10 - neg_pos_threshold, dtype=int)

        while num_pos_bins_left > 0:
            min_pos_bin_idx = np.argmin(pos_bins_counters)
            min_pos_bin_counter = pos_bins_counters[min_pos_bin_idx]

            num_samples_curr_bin = min(min_pos_bin_counter, num_pos_sample_left // num_pos_bins_left)
            num_pos_samples[min_pos_bin_idx] = num_samples_curr_bin

            num_pos_bins_left -= 1
            num_pos_sample_left -= num_samples_curr_bin
            pos_bins_counters[min_pos_bin_idx] = 2 ** 63 - 1

        for i in range(neg_pos_threshold, 10):
            bin_i = bins[i][np.random.choice(len(bins[i]), num_pos_samples[i - neg_pos_threshold], replace=False)]
            dist_norm_data_per_frame_pos = np.vstack((dist_norm_data_per_frame_pos, bin_i))

    # for i in range(len(bins)):
    #     if len(bins[i]) > max(int(max_bin_size * keep_ratio), 2, min_thresh) and i not in threshold:   # in case the bin size smaller than threshold
    #             bins[i] = bins[i][np.random.choice(len(bins[i]), max(int(max_bin_size * keep_ratio), 2, min_thresh), replace=False)]

    # print(dist_norm_data_per_frame_neg.shape)
    # print(dist_norm_data_per_frame_pos.shape)
    # print('-------------------------------------')

    dist_norm_data_per_frame = np.vstack((dist_norm_data_per_frame_neg, dist_norm_data_per_frame_pos))
    return dist_norm_data_per_frame

def normalize_data(ground_truth_mapping, max_pos_thresh=21, min_neg_thresh=7, neg_pos_threshold=3):
    size = int(np.sqrt(len(ground_truth_mapping)))
    dist_norm_data = np.empty((0, 4))

    for frame_idx in tqdm.tqdm(range(size)):
        ground_truth_mapping_current_frame = ground_truth_mapping[frame_idx*size:(frame_idx+1)*size, :]
        dist_norm_data_per_frame = normalize_data_per_frame(ground_truth_mapping_current_frame, max_pos_thresh,
                                                            min_neg_thresh, neg_pos_threshold)

        if len(dist_norm_data_per_frame) <= 1:
            raise f'Error! Frame {frame_idx} has sample size: {len(dist_norm_data_per_frame)}'

        dist_norm_data = np.vstack((dist_norm_data, dist_norm_data_per_frame))

    return dist_norm_data


if __name__ == '__main__':
    # load the ground truth data
    ground_truth_file_path = ('/home/vectr/Documents/Dataset/overlap_transformer/gt_overlap/royce_hall/'
                              'ground_truth_mapping_matrix.npy')
    ground_truth_mapping = np.load(ground_truth_file_path)

    # frame_idx = 226
    # size = int(np.sqrt(len(ground_truth_mapping)))
    # ground_truth_mapping_current_frame = ground_truth_mapping[frame_idx * size:(frame_idx + 1) * size, :]
    # normalize_data_per_frame(ground_truth_mapping_current_frame)

    a = normalize_data(ground_truth_mapping)
    print(a.shape)
