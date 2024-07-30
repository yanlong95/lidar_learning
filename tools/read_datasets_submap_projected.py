import os
import pathlib
import numpy as np
import torch
import einops

from tools.fileloader import read_image, load_xyz_rot, load_files


def overlaps_submaps_loader(overlaps_paths, submaps_paths, shuffle=True):
    overlaps_all = []
    anchor_submaps_all = []
    pos_neg_submaps_all = []

    for overlaps_path in overlaps_paths:
        overlaps = np.load(overlaps_path, allow_pickle=True)['overlaps']
        overlaps_all.extend(overlaps)

    for submaps_path in submaps_paths:
        anchor_submap_path = os.path.join(submaps_path, 'anchor.npy')
        pos_neg_submap_path = os.path.join(submaps_path, 'pos_neg.npy')
        anchor_submap = np.load(anchor_submap_path, allow_pickle=True)
        pos_neg_submap = np.load(pos_neg_submap_path, allow_pickle=True)
        anchor_submaps_all.extend(anchor_submap)
        pos_neg_submaps_all.extend(pos_neg_submap)

    overlaps_all = np.asarray(overlaps_all)
    anchor_submaps_all = np.asarray(anchor_submaps_all)
    pos_neg_submaps_all = np.asarray(pos_neg_submaps_all)

    if shuffle:
        mask = np.arange(overlaps_all.shape[0])
        np.random.shuffle(mask)
        overlaps_all = overlaps_all[mask]
        anchor_submaps_all = anchor_submaps_all[mask]
        pos_neg_submaps_all = pos_neg_submaps_all[mask]

    print(overlaps_all)
    print(anchor_submaps_all)
    print(pos_neg_submaps_all)
    print('=========================')

    return overlaps_all, anchor_submaps_all, pos_neg_submaps_all


def read_one_batch_overlaps_submap_one2n(img_folder_path, submaps_folder, overlaps, pos_neg_submaps, idx, channels,
                                         height, width, pos_max, neg_max, size_submap=2, device='cuda', shuffle=True):
    """
    Read a batch of training images (anchor, positive, negative). The anchor image is just a single image, but both
    positive and negative are submaps (consist with k images with).
    Args:
        overlaps: (numpy.ndarray) The table contain the information of overlaps. Each row corresponds to a sample. Col0
        is the anchor index. Col1 is the sequence index. Col2 is the overlap threshold for positive pair. Col3 is the
        number of positive pairs. Col4 is the number of negative pair. Col5-num_pos+5 are the positive pairs indices.
        Col105-num_neg+105 are the negative pairs indices. Col205-num_pos+205 are the positive pairs overlaps. Col305-
        num_neg+305 are the negative pairs overlaps.

        pos_neg_submaps: (numpy.ndarray) The table contain the information of submap. Each row corresponds to a sample
        scan. Each column is a keyframe index (in keyframe indices array). All the keyframes (columns) together form a
        submap.

        size_submap: (int) The number of keyframes in each submap.
    """

    batch = overlaps[idx, :]
    submaps = pos_neg_submaps[idx, :]
    submaps_base_path = pathlib.Path(submaps_folder[0]).parent

    # load query, positive, negative scans; number of positive, negative scans; sequence; overlap threshold; overlaps
    anchor_index = batch[0]                             # current frame index
    seq = batch[1]                                      # current frame sequence
    overlap_thresh = batch[2]                           # current frame overlap threshold (positive for > threshold)
    num_pos = batch[3]                                  # number of positive pairs for current scan
    num_neg = batch[4]                                  # number of negative pairs for current scan
    pos_samples_indices = batch[5:num_pos+5]            # indices of positive pairs for current scan
    neg_samples_indices = batch[105:num_neg+105]        # indices of negative pairs for current scan
    pos_samples_overlaps = batch[205:num_pos+205]       # overlap values of positive pairs for current scan
    neg_samples_overlaps = batch[305:num_neg+305]       # overlap values of negative pairs for current scan

    if shuffle:
        pos_shuffle_indices = np.arange(len(pos_samples_indices))
        neg_shuffle_indices = np.arange(len(neg_samples_indices))
        np.random.shuffle(pos_shuffle_indices)
        np.random.shuffle(neg_shuffle_indices)

        pos_samples_indices = pos_samples_indices[pos_shuffle_indices]
        neg_samples_indices = neg_samples_indices[neg_shuffle_indices]
        pos_samples_overlaps = pos_samples_overlaps[pos_shuffle_indices]
        neg_samples_overlaps = neg_samples_overlaps[neg_shuffle_indices]

    # constrain the batch size within max_size
    num_pos = min(batch[3], pos_max)
    num_neg = min(batch[4], neg_max)
    pos_samples_indices = pos_samples_indices[:num_pos]
    neg_samples_indices = neg_samples_indices[:num_neg]
    pos_samples_overlaps = pos_samples_overlaps[:num_pos]
    neg_samples_overlaps = neg_samples_overlaps[:num_neg]

    pos_samples_batch_overlaps = torch.zeros(num_pos, dtype=torch.float32).to(device)
    neg_samples_batch_overlaps = torch.zeros(num_neg, dtype=torch.float32).to(device)
    # anchor_submap_batch = torch.zeros(channels, height, width * (size_submap + 1), dtype=torch.float32).to(device)
    pos_samples_submap_batch = torch.zeros(num_pos, channels, height, width * (size_submap + 1), dtype=torch.float32).to(device)
    neg_samples_submap_batch = torch.zeros(num_neg, channels, height, width * (size_submap + 1), dtype=torch.float32).to(device)

    # load anchor tensor
    anchor_img_path = os.path.join(img_folder_path, seq, f'{str(anchor_index).zfill(6)}.png')
    anchor_batch = read_image(anchor_img_path, device)
    anchor_submap_batch = anchor_batch.repeat(1, 1, 1, size_submap + 1)
    # anchor_submap_batch = einops.repeat(anchor_batch, 'b c h w -> b c h (repeat w)', repeat=size_submap+1)

    # load positive tensors
    for i in range(num_pos):
        sample_idx = pos_samples_indices[i]
        img_path = os.path.join(img_folder_path, seq, f'{str(sample_idx).zfill(6)}.png')
        img_tensor = read_image(img_path, device).squeeze()     # in shape (1, H, W)
        pos_samples_batch_overlaps[i] = pos_samples_overlaps[i]

        # load keyframes for positive samples submap
        pos_samples_submap_batch[i, :, :, :width] = img_tensor
        for j in range(size_submap):
            img_path = os.path.join(submaps_base_path, seq, f'pos_neg/{str(anchor_index).zfill(6)}/{str(j)}.png')
            img_tensor = read_image(img_path, device).squeeze()  # in shape (1, H, W)
            pos_samples_submap_batch[i, :, :, width*(j+1):width*(j+2)] = img_tensor

    # load negative tensors
    for i in range(num_neg):
        sample_idx = neg_samples_indices[i]
        img_path = os.path.join(img_folder_path, seq, f'{str(sample_idx).zfill(6)}.png')
        img_tensor = read_image(img_path, device).squeeze()     # in shape (1, H, W)
        neg_samples_batch_overlaps[i] = neg_samples_overlaps[i]

        # load keyframes for negative samples submap
        neg_samples_submap_batch[i, :, :, :width] = img_tensor
        for j in range(size_submap):
            img_path = os.path.join(submaps_base_path, seq, f'pos_neg/{str(anchor_index).zfill(6)}/{str(j)}.png')
            img_tensor = read_image(img_path, device).squeeze()  # in shape (1, H, W)
            neg_samples_submap_batch[i, :, :, width*(j+1):width*(j+2)] = img_tensor

    # # load keyframes for anchor submap
    # for i in range(size_submap):
    #     img_path = os.path.join(img_kf_folder_path, seq, f'png_files/512/{str(submaps[idx, i]).zfill(6)}.png')
    #     img_tensor = read_image(img_path, device).squeeze()     # in shape (1, H, W)
    #     anchor_submap_batch[i, :, :, :] = img_tensor

    return (anchor_submap_batch, pos_samples_submap_batch, neg_samples_submap_batch, pos_samples_batch_overlaps,
            neg_samples_batch_overlaps, num_pos, num_neg)


def read_one_batch_overlaps_submap_n2n(img_folder_path, submaps_folder, overlaps, anchor_submaps, pos_neg_submaps,
                                       idx, channels, height, width, pos_max, neg_max, size_submap=2, device='cuda',
                                       shuffle=True):
    """
    Read a batch of training images (anchor, positive, negative). anchor, positive and negative are submaps (consist
    with k images with).
    Args:
        overlaps: (numpy.ndarray) The table contain the information of overlaps. Each row corresponds to a sample. Col0
        is the anchor index. Col1 is the sequence index. Col2 is the overlap threshold for positive pair. Col3 is the
        number of positive pairs. Col4 is the number of negative pair. Col5-num_pos+5 are the positive pairs indices.
        Col105-num_neg+105 are the negative pairs indices. Col205-num_pos+205 are the positive pairs overlaps. Col305-
        num_neg+305 are the negative pairs overlaps.

        submaps: (numpy.ndarray) The table contain the information of submap. Each row corresponds to a sample scan. Each
        column is a keyframe index (in keyframe indices array). All the keyframes (columns) together form a submap.

        size_submap: (int) The number of keyframes in each submap.
    """
    # TODO: check if normal batches are useless and delete them.

    batch = overlaps[idx, :]
    if isinstance(submaps_folder, list):
        submaps_base_path = pathlib.Path(submaps_folder[0]).parent
    elif isinstance(submaps_folder, str):
        submaps_base_path = pathlib.Path(submaps_folder).parent
    else:
        raise ValueError("Check the path of submaps folder.")

    # load query, positive, negative scans; number of positive, negative scans; sequence; overlap threshold; overlaps
    anchor_index = batch[0]                             # current frame iddex
    seq = batch[1]                                      # current frame sequence
    overlap_thresh = batch[2]                           # current frame overlap threshold (positive for > threshold)
    num_pos = batch[3]                                  # number of positive pairs for current scan
    num_neg = batch[4]                                  # number of negative pairs for current scan
    pos_samples_indices = batch[5:num_pos+5]            # indices of positive pairs for current scan
    neg_samples_indices = batch[105:num_neg+105]        # indices of negative pairs for current scan
    pos_samples_overlaps = batch[205:num_pos+205]       # overlap values of positive pairs for current scan
    neg_samples_overlaps = batch[305:num_neg+305]       # overlap values of negative pairs for current scan

    if shuffle:
        pos_shuffle_indices = np.arange(len(pos_samples_indices))
        neg_shuffle_indices = np.arange(len(neg_samples_indices))
        np.random.shuffle(pos_shuffle_indices)
        np.random.shuffle(neg_shuffle_indices)

        pos_samples_indices = pos_samples_indices[pos_shuffle_indices]
        neg_samples_indices = neg_samples_indices[neg_shuffle_indices]
        pos_samples_overlaps = pos_samples_overlaps[pos_shuffle_indices]
        neg_samples_overlaps = neg_samples_overlaps[neg_shuffle_indices]

    # constrain the batch size within max_size
    num_pos = min(batch[3], pos_max)
    num_neg = min(batch[4], neg_max)
    pos_samples_indices = pos_samples_indices[:num_pos]
    neg_samples_indices = neg_samples_indices[:num_neg]
    pos_samples_overlaps = pos_samples_overlaps[:num_pos]
    neg_samples_overlaps = neg_samples_overlaps[:num_neg]

    pos_samples_batch_overlaps = torch.zeros(num_pos, dtype=torch.float32).to(device)
    neg_samples_batch_overlaps = torch.zeros(num_neg, dtype=torch.float32).to(device)
    anchor_submap_batch = torch.zeros(1, channels, height, width * (size_submap + 1), dtype=torch.float32).to(device)
    pos_samples_submap_batch = torch.zeros(num_pos, channels, height, width * (size_submap + 1), dtype=torch.float32).to(device)
    neg_samples_submap_batch = torch.zeros(num_neg, channels, height, width * (size_submap + 1), dtype=torch.float32).to(device)

    # load anchor tensor
    anchor_img_path = os.path.join(img_folder_path, seq, f'{str(anchor_index).zfill(6)}.png')
    anchor_batch = read_image(anchor_img_path, device)
    anchor_submap_batch[:, :, :, :width] = anchor_batch.squeeze()
    for j in range(size_submap):
        img_path = os.path.join(submaps_base_path, seq, f'anchor/{str(anchor_index).zfill(6)}/{str(j)}.png')
        img_tensor = read_image(img_path, device).squeeze()  # in shape (1, H, W)
        anchor_submap_batch[:, :, :, width*(j+1):width*(j+2)] = img_tensor

    # load positive tensors
    for i in range(num_pos):
        sample_idx = pos_samples_indices[i]
        img_path = os.path.join(img_folder_path, seq, f'{str(sample_idx).zfill(6)}.png')
        img_tensor = read_image(img_path, device).squeeze()     # in shape (1, H, W)
        pos_samples_batch_overlaps[i] = pos_samples_overlaps[i]

        # load keyframes for positive samples submap
        pos_samples_submap_batch[i, :, :, :width] = img_tensor
        for j in range(size_submap):
            img_path = os.path.join(submaps_base_path, seq, f'pos_neg/{str(sample_idx).zfill(6)}/{str(j)}.png')
            img_tensor = read_image(img_path, device).squeeze()  # in shape (1, H, W)
            pos_samples_submap_batch[i, :, :, width*(j+1):width*(j+2)] = img_tensor

    # load negative tensors
    for i in range(num_neg):
        sample_idx = neg_samples_indices[i]
        img_path = os.path.join(img_folder_path, seq, f'{str(sample_idx).zfill(6)}.png')
        img_tensor = read_image(img_path, device).squeeze()     # in shape (1, H, W)
        neg_samples_batch_overlaps[i] = neg_samples_overlaps[i]

        # load keyframes for negative samples submap
        neg_samples_submap_batch[i, :, :, :width] = img_tensor
        for j in range(size_submap):
            img_path = os.path.join(submaps_base_path, seq, f'pos_neg/{str(sample_idx).zfill(6)}/{str(j)}.png')
            img_tensor = read_image(img_path, device).squeeze()  # in shape (1, H, W)
            neg_samples_submap_batch[i, :, :, width*(j+1):width*(j+2)] = img_tensor

    return (anchor_submap_batch, pos_samples_submap_batch, neg_samples_submap_batch, pos_samples_batch_overlaps,
            neg_samples_batch_overlaps, num_pos, num_neg)


if __name__ == '__main__':
    frame_poses_path = '/media/vectr/vectr3/Dataset/overlap_transformer/poses/botanical_garden/poses.txt'
    keyframes_poses_path = '/media/vectr/vectr3/Dataset/overlap_transformer/keyframes/botanical_garden/poses/poses_kf.txt'
    img_folder_path = '/media/vectr/vectr3/Dataset/overlap_transformer/png_files/512'
    img_kf_folder_path = '/media/vectr/vectr3/Dataset/overlap_transformer/keyframes'
    overlaps_folder = '/media/vectr/vectr3/Dataset/overlap_transformer/gt_overlaps'
    submaps_folder = '/media/vectr/vectr3/Dataset/overlap_transformer/submaps/overlap'
    seqs =  ["bomb_shelter", "botanical_garden", "bruin_plaza", "court_of_sciences", "dickson_court", "geo_loop",
             "kerckhoff", "luskin", "royce_hall", "sculpture_garden"]
    overlaps_paths = [os.path.join(overlaps_folder, seq, 'overlaps_train.npz') for seq in seqs]
    submaps_paths = [os.path.join(submaps_folder, seq) for seq in seqs]

    xyz, _ = load_xyz_rot(frame_poses_path)
    xyz_kf, _ = load_xyz_rot(keyframes_poses_path)

    overlaps, anchor_submaps, pos_neg_submaps = overlaps_submaps_loader(overlaps_paths, submaps_paths, shuffle=True)

    # anchor, pos, neg, _, _, _, _ = read_one_batch_overlaps_submap_one2n(img_folder_path, submaps_folder, overlaps, pos_neg_submaps, 1000, 1, 32, 512, 10, 10,
    #                                                                     device='cpu')
    anchor_, pos_, neg_, _, _, _, _ = read_one_batch_overlaps_submap_n2n(img_folder_path, submaps_paths, overlaps, anchor_submaps, pos_neg_submaps, 0, 1, 32, 512, 10, 10,
                                                                         device='cpu')
