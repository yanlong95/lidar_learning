import os
import tqdm
import faiss
import yaml
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from modules.overlap_transformer import OverlapTransformer32
from modules.losses.overlap_transformer_loss import triplet_confidence_loss
from tools.fileloader import load_files, load_xyz_rot, load_overlaps, read_image
from tools.read_datasets import read_one_batch_overlaps
from tools.utils import RunningAverage, load_checkpoint
from tools.utils_func import compute_top_k_keyframes


def validation(model, top_n=5, metric='euclidean', method='overlap', dist_thresh=5.0):
    """
    Validation function for the overlap transformer model.
    Args:
        model: (nn.module) the pytorch model.
        top_n: (int) the number of top keyframes to search.
        metric: (string) the metric to search the top keyframes (euclidean or cosine).
        method: (string) the method to validate the model (euclidean or overlap).
    Returns:
        loss: (float) the loss of the model.
        recall: (float) the recall of the model.
    """
    # ===============================================================================
    # loading paths and parameters
    config_filename = '/home/vectr/PycharmProjects/lidar_learning/configs/config.yml'
    params_filename = '/home/vectr/PycharmProjects/lidar_learning/configs/parameters.yml'
    config = yaml.safe_load(open(config_filename))
    params = yaml.safe_load(open(params_filename))

    img_folder = config['data_root']['png_files']                                       # img path for all frames
    overlaps_folder = config['data_root']['gt_overlaps']
    overlaps_table_folder = config['data_root']['overlaps']
    poses_folder = config['data_root']['poses']
    keyframes_folder = config['data_root']['keyframes']

    valid_seq = config['seqs']['valid'][0]
    channels = params['learning']['channels']
    height = params['learning']['height']
    width = params['learning']['width']
    margin1 = params['learning']['margin1']
    alpha = params['learning']['alpha']
    num_pos_max = params['learning']['num_pos_max']
    num_neg_max = params['learning']['num_neg_max']
    # ===============================================================================
    valid_img_folder = os.path.join(img_folder, valid_seq)                       # img path for validation frames
    valid_img_kf_folder = os.path.join(keyframes_folder, valid_seq, 'png_files/512')    # img path for keyframes

    poses_paths = os.path.join(poses_folder, valid_seq, 'poses.txt')
    poses_kf_paths = os.path.join(keyframes_folder, valid_seq, 'poses/poses_kf.txt')
    valid_overlaps_path = os.path.join(overlaps_folder, valid_seq, 'overlaps_full.npz')
    valid_overlaps_table_path = os.path.join(overlaps_table_folder, f'{valid_seq}.bin')

    # load files
    xyz, _ = load_xyz_rot(poses_paths)
    xyz_kf, _ = load_xyz_rot(poses_kf_paths)
    img_paths = load_files(valid_img_folder)
    img_kf_paths = load_files(valid_img_kf_folder)
    overlaps = load_overlaps(valid_overlaps_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    assert metric == 'euclidean' or metric == 'cosine', "Unknown metric! Must be 'euclidean' or 'cosine'!"
    assert method == 'euclidean' or method == 'overlap', "Unknown method! Must be 'euclidean' or 'overlap'!"

    # calculate descriptors and searching
    with torch.no_grad():
        model.eval()
        num_scans = len(img_paths)
        num_scans_kf = len(img_kf_paths)

        # -----------------------------------------------------------------------------------------------
        # compute loss for validation set
        loss_avg = RunningAverage()
        for i in tqdm.tqdm(range(num_scans)):
            # read batch
            anchor_batch, pos_batch, neg_batch, pos_overlaps, neg_overlaps, num_pos, num_neg = read_one_batch_overlaps(
                img_folder, overlaps, i, channels, height, width, num_pos_max, num_neg_max, device, shuffle=False)

            # in case no pair
            if num_pos == 0 or num_neg == 0:
                continue

            input_batch = torch.cat((anchor_batch, pos_batch, neg_batch), dim=0)
            output_batch = model(input_batch)
            o1, o2, o3 = torch.split(output_batch, [1, num_pos, num_neg], dim=0)
            loss = triplet_confidence_loss(o1, o2, o3, pos_overlaps, margin1, alpha=alpha, lazy=False,
                                           ignore_zero_loss=True, metric=metric)
            # in case no valid loss
            if loss == -1:
                continue
            else:
                loss_avg.update(loss.item())

        # -----------------------------------------------------------------------------------------------
        # calculate all descriptors
        descriptors = np.zeros((num_scans, 256), dtype='float32')
        for i in tqdm.tqdm(range(num_scans)):
            # load a scan
            current_batch = read_image(img_paths[i], device)
            current_batch = torch.cat((current_batch, current_batch), dim=0)        # no idea why, keep it now, need to test and remove !!!!!

            # calculate descriptor
            current_descriptor = model(current_batch)
            descriptors[i, :] = current_descriptor[0, :].cpu().detach().numpy()
        descriptors = descriptors.astype('float32')

        # calculate keyframe descriptors
        descriptors_kf = np.zeros((num_scans_kf, 256), dtype='float32')
        for i in range(num_scans_kf):
            current_batch = read_image(img_kf_paths[i], device)
            current_batch = torch.cat((current_batch, current_batch), dim=0)        # no idea why, keep it now, need to test and remove !!!!!

            current_descriptor = model(current_batch)
            descriptors_kf[i, :] = current_descriptor[0, :].cpu().detach().numpy()
        descriptors_kf = descriptors_kf.astype('float32')

        # -----------------------------------------------------------------------------------------------
        # searching (all scans and keyframes)
        d = descriptors.shape[1]
        index_kf = faiss.IndexFlatL2(d) if metric == 'euclidean' else faiss.IndexFlatIP(d)

        # compute closest keyframes for validation set based on the prediction
        index_kf.add(descriptors_kf)
        _, top_n_keyframes_pred = index_kf.search(descriptors, top_n)

        # calculate the closest keyframe based on the distances or overlap values (top 1 ground truth)
        top_n_keyframes = compute_top_k_keyframes(poses_paths, poses_kf_paths, valid_overlaps_table_path, top_k=1,
                                                  metric=method)

        num_pos_pred = 0
        num_pos_pred_within_dist_threshold = 0
        for i in range(num_scans):
            # check if any predicted keyframes is the true closest keyframe
            top_n_kf_ground_truth = top_n_keyframes[i, 0]
            top_n_kf_prediction = top_n_keyframes_pred[i, :]
            if top_n_kf_ground_truth in top_n_kf_prediction:
                num_pos_pred += 1

            # check if any of the current frame and the predicted keyframes is within the distance threshold
            dists = np.linalg.norm(xyz_kf[top_n_kf_prediction, :] - xyz[i, :], axis=1)
            if np.any(dists < dist_thresh):
                num_pos_pred_within_dist_threshold += 1

        # # plot view
        # for i in range(num_scans):
        #     plt.clf()
        #     plt.scatter(xyz[:, 0], xyz[:, 1], color='blue')
        #     plt.scatter(xyz_kf[:, 0], xyz_kf[:, 1], color='violet')
        #     plt.scatter(xyz[i, 0], xyz[i, 1], color='gold')
        #     plt.scatter(xyz_kf[top_n_keyframes_pred[i], 0], xyz_kf[top_n_keyframes_pred[i], 1], color='green')
        #     plt.show(block=False)
        #     plt.pause(0.01)

        # calculate the average loss and recall
        avg_loss = loss_avg()
        recall = num_pos_pred / num_scans
        recall_within_dist_threshold = num_pos_pred_within_dist_threshold / num_scans
        return avg_loss, recall, recall_within_dist_threshold


if __name__ == '__main__':
    weights_path = '/media/vectr/vectr3/Dataset/overlap_transformer/weights/weights_06_19/best.pth.tar'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OverlapTransformer32(height=32, width=512, channels=1, use_transformer=True).to(device)
    load_checkpoint(weights_path, model)
    l, r, rd = validation(model, metric='cosine', method='overlap')
    print(f'loss: {l}')
    print(f'recall: {r}')
    print(f'recall_within_dist_threshold: {rd}')
    # cosine has lower recall than euclidean, but margin is harder, not sure which one take the control.
