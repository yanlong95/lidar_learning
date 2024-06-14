import os
import numpy as np
import torch

from tools.fileloader import read_image, load_xyz_rot


def overlaps_loader(overlaps_paths, shuffle=True):
    overlaps_all = []
    for overlaps_path in overlaps_paths:
        overlaps = np.load(overlaps_path, allow_pickle=True)['overlaps']
        overlaps_all.extend(overlaps)

    overlaps_all = np.asarray(overlaps_all)
    if shuffle:
        np.random.shuffle(overlaps_all)

    return overlaps_all


def read_one_batch_pos_neg(img_folder_path, overlaps, idx, channels, height, width, pos_max, neg_max, device='cuda',
                           shuffle=True):       # without end

    batch = overlaps[idx, :]

    # load query, positive, negative scans; number of positive, negative scans; sequence; overlap threshold
    anchor_index = batch[0]
    seq = batch[1]
    overlap_thresh = batch[2]
    num_pos = batch[3]
    num_neg = batch[4]
    pos_samples_indices = batch[5:num_pos+5]
    neg_samples_indices = batch[105:num_neg+105]

    if shuffle:
        np.random.shuffle(pos_samples_indices)
        np.random.shuffle(neg_samples_indices)

    # constrain the batch size within max_size
    num_pos = min(num_pos, pos_max)
    num_neg = min(num_neg, neg_max)
    pos_samples_indices = pos_samples_indices[:num_pos]
    neg_samples_indices = neg_samples_indices[:num_neg]

    pos_sample_batch = torch.zeros(num_pos, channels, height, width, dtype=torch.float32).to(device)
    neg_sample_batch = torch.zeros(num_neg, channels, height, width, dtype=torch.float32).to(device)

    # load anchor tensor
    anchor_img_path = os.path.join(img_folder_path, seq, f'{str(anchor_index).zfill(6)}.png')
    anchor_batch = read_image(anchor_img_path, device)

    # load positive tensors
    for i in range(num_pos):
        img_path = os.path.join(img_folder_path, seq, f'{str(pos_samples_indices[i]).zfill(6)}.png')
        img_tensor = read_image(img_path, device).squeeze()     # in shape (1, H, W)
        pos_sample_batch[i, :, :, :] = img_tensor

    # load negative tensors
    for i in range(num_neg):
        img_path = os.path.join(img_folder_path, seq, f'{str(neg_samples_indices[i]).zfill(6)}.png')
        img_tensor = read_image(img_path, device).squeeze()     # in shape (1, H, W)
        neg_sample_batch[i, :, :, :] = img_tensor

    return anchor_batch, pos_sample_batch, neg_sample_batch, num_pos, num_neg


def read_one_batch_pos_neg_numerical(img_folder_path, overlaps, idx, channels, height, width, pos_max, neg_max,
                                     device='cuda', shuffle=True):

    batch = overlaps[idx, :]

    # load query, positive, negative scans; number of positive, negative scans; sequence; overlap threshold; overlaps
    anchor_index = batch[0]                             # current frame iddex
    seq = batch[1]                                      # current frame sequence
    overlap_thresh = batch[2]                           # current frame overlap threshold (positive for > threshold)
    num_pos = batch[3]                                  # number of positive pairs for current scan
    num_neg = batch[4]                                  # number of negative pairs for current scan
    pos_samples_indices = batch[5:num_pos+5]            # indices of positive pairs for current scan
    neg_samples_indices = batch[105:num_neg+105]        # indices of negative pairs for current scan
    pos_sample_overlaps = batch[205:num_pos+205]        # overlap values of positive pairs for current scan
    neg_sample_overlaps = batch[305:num_neg+305]        # overlap values of negative pairs for current scan

    if shuffle:
        pos_shuffle_indices = np.arange(len(pos_samples_indices))
        neg_shuffle_indices = np.arange(len(neg_samples_indices))
        np.random.shuffle(pos_shuffle_indices)
        np.random.shuffle(neg_shuffle_indices)

        pos_samples_indices = pos_samples_indices[pos_shuffle_indices]
        neg_samples_indices = neg_samples_indices[neg_shuffle_indices]
        pos_sample_overlaps = pos_sample_overlaps[pos_shuffle_indices]
        neg_sample_overlaps = neg_sample_overlaps[neg_shuffle_indices]

    # constrain the batch size within max_size
    num_pos = min(batch[3], pos_max)
    num_neg = min(batch[4], neg_max)
    pos_samples_indices = pos_samples_indices[:num_pos]
    neg_samples_indices = neg_samples_indices[:num_neg]
    pos_sample_overlaps = pos_sample_overlaps[:num_pos]
    neg_sample_overlaps = neg_sample_overlaps[:num_neg]

    pos_sample_batch = torch.zeros(num_pos, channels, height, width, dtype=torch.float32).to(device)
    neg_sample_batch = torch.zeros(num_neg, channels, height, width, dtype=torch.float32).to(device)
    pos_sample_batch_overlaps = torch.zeros(num_pos, dtype=torch.float32).to(device)
    neg_sample_batch_overlaps = torch.zeros(num_neg, dtype=torch.float32).to(device)

    # load anchor tensor
    anchor_img_path = os.path.join(img_folder_path, seq, f'{str(anchor_index).zfill(6)}.png')
    anchor_batch = read_image(anchor_img_path, device)

    # load positive tensors
    for i in range(num_pos):
        img_path = os.path.join(img_folder_path, seq, f'{str(pos_samples_indices[i]).zfill(6)}.png')
        img_tensor = read_image(img_path, device).squeeze()     # in shape (1, H, W)
        pos_sample_batch[i, :, :, :] = img_tensor
        pos_sample_batch_overlaps[i] = pos_sample_overlaps[i]

    # load negative tensors
    for i in range(num_neg):
        img_path = os.path.join(img_folder_path, seq, f'{str(neg_samples_indices[i]).zfill(6)}.png')
        img_tensor = read_image(img_path, device).squeeze()     # in shape (1, H, W)
        neg_sample_batch[i, :, :, :] = img_tensor
        neg_sample_batch_overlaps[i] = neg_sample_overlaps[i]

    return (anchor_batch, pos_sample_batch, neg_sample_batch, pos_sample_batch_overlaps, neg_sample_batch_overlaps,
            num_pos, num_neg)


if __name__ == '__main__':
    train_dataset = ['/media/vectr/T9/Dataset/overlap_transformer/gt_overlaps/botanical_garden/overlaps_train.npz',
                     '/media/vectr/T9/Dataset/overlap_transformer/gt_overlaps/botanical_garden/overlaps_valid.npz']
    img_folder_path = '/media/vectr/T9/Dataset/overlap_transformer/png_files/900'
    overlaps = overlaps_loader(train_dataset, shuffle=True)
    read_one_batch_pos_neg(img_folder_path, overlaps, 1000, 1, 32, 900)
