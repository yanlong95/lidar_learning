import os
import numpy as np
import torch
import cv2

from tools.fileloader import read_image


def overlaps_loader(overlaps_paths, shuffle=True):
    overlaps_all = []
    for overlaps_path in overlaps_paths:
        overlaps = np.load(overlaps_path, allow_pickle=True)['overlaps']
        overlaps_all.extend(overlaps)

    overlaps_all = np.asarray(overlaps_all)
    if shuffle:
        np.random.shuffle(overlaps_all)

    return overlaps_all


def read_one_batch_pos_neg(img_folder_path, overlaps, idx, shuffle=True):  # without end
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    pos_sample_batch = torch.zeros(num_pos, 1, 32, 900, dtype=torch.float32).to(device)
    neg_sample_batch = torch.zeros(num_neg, 1, 32, 900, dtype=torch.float32).to(device)

    # load anchor tensor
    anchor_img_path = os.path.join(img_folder_path, seq, f'{str(anchor_index).zfill(6)}.png')
    anchor_batch = read_image(anchor_img_path)

    # load positive tensors
    for i in range(len(pos_samples_indices)):
        img_path = os.path.join(img_folder_path, seq, f'{str(pos_samples_indices[i]).zfill(6)}.png')
        img_tensor = read_image(img_path).squeeze()     # in shape (1, H, W)
        pos_sample_batch[i, :, :, :] = img_tensor

    # load negative tensors
    for i in range(len(neg_samples_indices)):
        img_path = os.path.join(img_folder_path, seq, f'{str(neg_samples_indices[i]).zfill(6)}.png')
        img_tensor = read_image(img_path).squeeze()     # in shape (1, H, W)
        neg_sample_batch[i, :, :, :] = img_tensor

    return anchor_batch, pos_sample_batch, neg_sample_batch, num_pos, num_neg


def read_one_need_from_seq(folder, seq, file):
    img_path = os.path.join(folder, '900', seq, f'{file}.png')
    return read_image(img_path)


if __name__ == '__main__':
    train_dataset = ['/media/vectr/T9/Dataset/overlap_transformer/gt_overlaps/botanical_garden/overlaps_train.npz',
                     '/media/vectr/T9/Dataset/overlap_transformer/gt_overlaps/botanical_garden/overlaps_valid.npz']
    img_folder_path = '/media/vectr/T9/Dataset/overlap_transformer/png_files/900'
    overlaps = overlaps_loader(train_dataset, shuffle=True)
    read_one_batch_pos_neg(img_folder_path, overlaps, 1000)
