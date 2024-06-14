import os
import tqdm
import faiss
import yaml
import torch
import numpy as np
from modules.overlap_transformer import OverlapTransformer32
from modules.losses.overlap_transformer_loss import mean_squared_error_loss
from tools.fileloader import load_files, read_image, load_xyz_rot


def validation(model, top_n=5):
    # ===============================================================================
    # loading paths and parameters
    config_filename = '/home/vectr/PycharmProjects/lidar_learning/configs/config.yml'
    config = yaml.safe_load(open(config_filename))
    valid_scans_folder = config['data_root']['png_files']
    ground_truth_folder = config['data_root']['gt_overlaps']
    valid_seq = config['seqs']['valid'][0]

    poses_folder = config['data_root']['poses']
    keyframes_folder = config['data_root']['keyframes']
    # ===============================================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    valid_scan_paths = load_files(os.path.join(valid_scans_folder, '512', valid_seq))
    ground_truth_paths = os.path.join(ground_truth_folder, valid_seq, 'overlaps.npz')
    ground_truth_overlaps = np.load(ground_truth_paths)['arr_0']

    # find the closest top_n keyframes for each scan
    poses_paths = os.path.join(poses_folder, valid_seq, 'poses.txt')
    poses_kf_paths = os.path.join(keyframes_folder, valid_seq, 'poses/poses_kf.txt')
    img_kf_paths = load_files(os.path.join(keyframes_folder, valid_seq, 'png_files/512'))

    xyz, _ = load_xyz_rot(poses_paths)
    xyz_kf, _ = load_xyz_rot(poses_kf_paths)

    index_dists = faiss.IndexFlatL2(3)
    index_dists.add(xyz_kf)
    D_dists, I_dists = index_dists.search(xyz, top_n)

    # calculate descriptors and searching
    with torch.no_grad():
        num_scans = len(valid_scan_paths)
        num_kf = len(xyz_kf)

        # calculate all descriptors
        descriptors = np.zeros((num_scans, 256))
        for i in tqdm.tqdm(range(num_scans)):
            # load a scan
            current_batch = read_image(valid_scan_paths[i], device)
            current_batch = torch.cat((current_batch, current_batch), dim=0)        # no idea why, keep it now

            # calculate descriptor
            model.eval()
            current_descriptor = model(current_batch)
            descriptors[i, :] = current_descriptor[0, :].cpu().detach().numpy()
        descriptors = descriptors.astype('float32')

        # calculate keyframe descriptors
        descriptors_kf = np.zeros((num_kf, 256))
        for i in range(num_kf):
            current_batch = read_image(img_kf_paths[i], device)
            current_batch = torch.cat((current_batch, current_batch), dim=0)        # no idea why, keep it now

            model.eval()
            current_descriptor = model(current_batch)
            descriptors_kf[i, :] = current_descriptor[0, :].cpu().detach().numpy()
        descriptors_kf = descriptors_kf.astype('float32')

        # searching (all scans and keyframes)
        d = descriptors.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(descriptors)

        index_kf = faiss.IndexFlatL2(d)
        index_kf.add(descriptors_kf)

        # search the closest descriptors for each one in the validation set
        """
        2 ways to calculate the validation.
        a. search the closest descriptors, if their overlap values greater than the threshold, then positive prediction
           (not recommend as most of top_n scans are close to current scan even for a random model. choose a large top_n
           if you want to use this way).
        b. search top_n positive and negative descriptors, if distances between the all positive descriptors are smaller
           than the negative descriptors, then positive prediction.
        c. add another valid method. choose the top n closest keyframes based on the distance of descriptors, if any 
           chosen keyframe is the closest keyframe based on global distance, then a correct prediction.
        d. compute the losses for all pairs (MSE between estimated overlaps and true overlaps).
        """

        num_pos_pred = 0
        # method = 'overlap_thresh'         # a
        method = 'pos_neg_descriptors'      # b
        # method = 'closest_keyframe'       # c
        # method = 'overlap'                # d

        for i in range(num_scans):
            ground_truth = ground_truth_overlaps[i]
            pos_scans = ground_truth[ground_truth[:, 2] >= ground_truth[:, 3]]
            neg_scans = ground_truth[ground_truth[:, 2] < ground_truth[:, 3]]

            # in case no enough positive scans
            top_n = min(len(pos_scans), top_n)

            if method == 'overlap_thresh':
                D, I = index.search(descriptors[i, :].reshape(1, -1), top_n)

                if I[:, 0] == i:
                    min_index = I[:, 1:]
                else:
                    min_index = I[:, :]

                if np.all(ground_truth[min_index, 2] > ground_truth[min_index, 3]):
                    num_pos_pred += 1
            elif method == 'pos_neg_descriptors':
                pos_indices = np.random.choice(pos_scans[:, 1], top_n, replace=False).astype(int)
                neg_indices = np.random.choice(neg_scans[:, 1], top_n, replace=False).astype(int)

                pos_descriptors = descriptors[pos_indices, :]
                neg_descriptors = descriptors[neg_indices, :]

                pos_dists = np.linalg.norm(pos_descriptors - descriptors[i, :], axis=1)
                neg_dists = np.linalg.norm(neg_descriptors - descriptors[i, :], axis=1)

                num_pos_pred_j = 0
                for j in range(top_n):
                    if np.all(pos_dists[j] - neg_dists <= 0):
                        num_pos_pred_j += 1
                if num_pos_pred_j == top_n:
                    num_pos_pred += 1
            elif method == 'closest_keyframe':
                D_kf, I_kf = index_kf.search(descriptors[i, :].reshape(1, -1), top_n)
                min_index_kf = I_kf[:, :]
                min_index_dists = I_dists[i, :]
                intersection = np.intersect1d(min_index_kf, min_index_dists)
                if intersection.size > 0:       # change to top_n // 2 for hard mode
                    num_pos_pred += 1
            else:
                pos_indices = pos_scans[:, 1].astype(int)
                neg_indices = np.random.choice(neg_scans[:, 1], len(pos_scans), replace=False).astype(int)  # balance
                anchor_tensor = torch.tensor(descriptors[i, :].reshape(1, -1)).to(device)
                pos_tensors = torch.tensor(descriptors[pos_indices, :]).to(device)
                neg_tensors = torch.tensor(descriptors[neg_indices, :]).to(device)
                pos_overlaps = torch.from_numpy(ground_truth[pos_indices, 2]).to(device)
                neg_overlaps = torch.from_numpy(ground_truth[neg_indices, 2]).to(device)
                loss = mean_squared_error_loss(anchor_tensor, pos_tensors, neg_tensors, pos_overlaps, neg_overlaps,
                                               alpha=1.0)
                print(f'loss: {loss.item()}')
                return loss.item()


    # precision = num_pos_pred / (top_n * num_valid)
    precision = num_pos_pred / num_scans
    # precision_neg = num_neg_pred / num_valid
    print(f'top {top_n} precision: {precision}.\n')
    # print(f'top {top_n} precision_neg: {precision_neg}')
    return precision


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OverlapTransformer32(height=32, width=900, channels=1, use_transformer=True).to(device)
    validation(model)
