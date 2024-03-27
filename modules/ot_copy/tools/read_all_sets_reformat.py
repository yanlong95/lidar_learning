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


def read_one_batch_pos_neg(data_root_folder, f1_index, f1_seq, train_imgf1, train_imgf2, train_dir1, train_dir2,
                           train_overlap, overlap_thresh, overlaps, idx, shuffle=True):  # without end
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch = overlaps[idx, :]

    # load query, positive, negative scans; number of positive, negative scans; sequence; overlap threshold
    anchor = batch[0]
    seq = batch[1]
    overlap_thresh = batch[2]
    num_pos = batch[3]
    num_neg = batch[4]
    pos_samples = batch[5:num_pos+5]
    neg_samples = batch[105:num_neg+105]

    batch_size = num_pos + num_neg + 1
    sample_batch = torch.zeros(batch_size, 1, 32, 900, dtype=torch.float32).to(device)
    sample_truth = torch.zeros(batch_size, 1, dtype=torch.float32).to(device)

    if shuffle:
        pos_samples = np.random.shuffle(pos_samples)
        neg_samples = np.random.shuffle(neg_samples)





    batch_size = 0
    for tt in range(len(train_imgf1)):
        if f1_index == train_imgf1[tt] and f1_seq == train_dir1[tt]:
            batch_size = batch_size + 1

    sample_batch = torch.from_numpy(np.zeros((batch_size, 1, 32, 900))).type(torch.FloatTensor).cuda()
    sample_truth = torch.from_numpy(np.zeros((batch_size, 1))).type(torch.FloatTensor).cuda()

    pos_idx = 0
    neg_idx = 0
    pos_num = 0
    neg_num = 0

    for j in range(len(train_imgf1)):
        pos_flag = False
        if f1_index == train_imgf1[j] and f1_seq==train_dir1[j]:
            if train_overlap[j] > overlap_thresh[j]:
                pos_num = pos_num + 1
                pos_flag = True
            else:
                neg_num = neg_num + 1

            img_path = os.path.join(data_root_folder, '900', train_dir2[j], f'{train_imgf2[j]}.png')
            depth_data_r = np.array(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))

            depth_data_tensor_r = torch.from_numpy(depth_data_r).type(torch.FloatTensor).cuda()
            depth_data_tensor_r = torch.unsqueeze(depth_data_tensor_r, dim=0)

            if pos_flag:
                sample_batch[pos_idx,:,:,:] = depth_data_tensor_r
                sample_truth[pos_idx, :] = torch.from_numpy(np.array(train_overlap[j])).type(torch.FloatTensor).cuda()
                pos_idx = pos_idx + 1
            else:
                sample_batch[batch_size-neg_idx-1, :, :, :] = depth_data_tensor_r
                sample_truth[batch_size-neg_idx-1, :] = torch.from_numpy(np.array(train_overlap[j])).type(torch.FloatTensor).cuda()
                neg_idx = neg_idx + 1


    return sample_batch, sample_truth, pos_num, neg_num


def read_one_need_from_seq(folder, seq, file):
    img_path = os.path.join(folder, '900', seq, f'{file}.png')
    return read_image(img_path)


if __name__ == '__main__':
    train_dataset = ['/Users/yanlong/Desktop/gt_overlaps/botanical_garden/overlaps_train.npz',
                     '/Users/yanlong/Desktop/gt_overlaps/botanical_garden/overlaps_valid.npz']
    overlaps_loader(train_dataset, shuffle=True)
