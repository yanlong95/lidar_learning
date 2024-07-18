import os
import yaml
import numpy as np
import torch
import faiss
import tqdm
import matplotlib
import matplotlib.pyplot as plt

from modules.overlap_transformer_submap import OverlapTransformer32Submap
from tools.fileloader import load_files, load_xyz_rot, load_overlaps, load_descriptors, read_image, load_submaps
from tools.utils_func import compute_top_k_keyframes


class testHandler():
    def __init__(self, params, img_folder, img_kf_folder, frames_poses_path, keyframes_poses_path, weights_path,
                 overlaps_table_path=None, top_n=5, skip=1, method='overlap', load_descriptors=False,
                 descriptors_folder=None, predictions_folder=None, submaps_path=None):
        # parameters
        self.params = params
        self.height = params['learning']['height']
        self.width = params['learning']['width']
        self.channels = params['learning']['channels']
        self.descriptor_size = params['learning']['descriptor_size']
        self.dist_thresh = params['learning']['dist_thresh']
        self.metric = params['learning']['metric']
        self.submap_size = params['learning']['submap_size']
        self.top_n = top_n
        self.skip = skip
        self.method = method
        self.load_descriptors = load_descriptors

        # model and device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = OverlapTransformer32Submap(self.height, self.width, self.channels, self.submap_size).to(self.device)

        # paths
        self.img_folder = img_folder
        self.img_kf_folder = img_kf_folder
        self.overlaps_table_path = overlaps_table_path
        self.frames_poses_path = frames_poses_path
        self.keyframes_poses_path = keyframes_poses_path
        self.weights_path = weights_path
        self.descriptors_folder = descriptors_folder
        self.predictions_folder = predictions_folder
        self.submaps_folder = submaps_path

    def compute_descriptors(self, img_paths, model, submaps, submap_size):
        num_scans = len(img_paths)
        descriptors = np.zeros((num_scans, self.descriptor_size), dtype=np.float32)

        model.eval()
        with torch.no_grad():
            for i in tqdm.tqdm(range(num_scans)):
                curr_batch = read_image(img_paths[i])
                # curr_batch = torch.cat((curr_batch, curr_batch), dim=0)     # no idea why, keep it now, need to test and remove !!!!!
                for j in range(submap_size):
                    submap_scan = read_image(img_paths[submaps[i, j]])
                    curr_batch = torch.cat((curr_batch, submap_scan), dim=-1)
                curr_descriptors = model(curr_batch)
                descriptors[i, :] = curr_descriptors[0, :].cpu().detach().numpy()

        descriptors = descriptors.astype(np.float32)
        return descriptors

    def compute_predictions(self, descriptors, descriptors_kf):
        d = descriptors.shape[1]
        index_kf = faiss.IndexFlatL2(d) if self.metric == 'euclidean' else faiss.IndexFlatIP(d)
        index_kf.add(descriptors_kf)
        _, top_k_n_keyframes_pred = index_kf.search(descriptors, self.top_n)
        return top_k_n_keyframes_pred

    def compute_recall(self, top_n_keyframes, top_n_keyframes_pred, xyz, xyz_kf):
        num_scans = len(top_n_keyframes_pred)

        num_pos_pred = 0
        num_pos_pred_within_dist_threshold = 0

        pos_pred_indices = []                   # indices for positive predictions
        neg_pred_indices = []                   # indices for negative predictions
        pos_pred_kf_indices = []                # indices for positive predicted keyframes
        neg_pred_kf_indices = []                # indices for negative predicted keyframes

        for i in range(num_scans):
            # check if any predicted keyframes is the true closest keyframe
            top_n_kf_ground_truth = top_n_keyframes[i, 0]
            top_n_kf_prediction = top_n_keyframes_pred[i, :]
            if top_n_kf_ground_truth in top_n_kf_prediction:
                num_pos_pred += 1
                pos_pred_indices.append(i)
                pos_pred_kf_indices.append(top_n_kf_ground_truth)
            else:
                neg_pred_indices.append(i)
                neg_pred_kf_indices.append(top_n_kf_ground_truth)

            # check if any of ||current frame - predicted keyframes|| is within the distance threshold (for further use)
            dists = np.linalg.norm(xyz_kf[top_n_kf_prediction, :] - xyz[i, :], axis=1)
            if np.any(dists < self.dist_thresh):
                num_pos_pred_within_dist_threshold += 1

        recall = num_pos_pred / num_scans
        pos_pred_indices = np.array(pos_pred_indices)
        neg_pred_indices = np.array(neg_pred_indices)
        pos_pred_kf_indices = np.array(pos_pred_kf_indices)
        neg_pred_kf_indices = np.array(neg_pred_kf_indices)
        return recall, pos_pred_indices, neg_pred_indices, pos_pred_kf_indices, neg_pred_kf_indices

    def compute_confidence_scores(self, descriptors, descriptors_kf, top_n_keyframes, use_min=True):
        num_scans = len(descriptors)
        confidence_scores = np.zeros(num_scans)

        for i in range(num_scans):
            curr_descriptor = descriptors[i, :]
            top_n_keyframe = top_n_keyframes[i, :]
            top_n_descriptors = descriptors_kf[top_n_keyframe, :]

            if self.metric == 'euclidean':
                diff = np.linalg.norm(curr_descriptor - top_n_descriptors, axis=1)
            else:
                cosine = np.sum(curr_descriptor * top_n_descriptors, axis=1)
                cosine = np.clip(cosine, -1, 1)
                diff = np.arccos(cosine)

            if use_min:
                top_n_best_dist = np.min(diff)
            else:
                top_n_best_dist = np.mean(diff)
            confidence_scores[i] = top_n_best_dist

        confidence_scores /= np.max(confidence_scores)
        confidence_scores = np.ones_like(confidence_scores) - confidence_scores

        return confidence_scores

    def visualization(self, pos_pred_indices, neg_pred_indices, xyz, xyz_kf, confidence_scores, predictions_path=None):
        if len(pos_pred_indices) > 0:
            positive_points = xyz[pos_pred_indices, :]
        if len(neg_pred_indices) > 0:
            negative_points = xyz[neg_pred_indices, :]

        fig, [ax1, ax2] = plt.subplots(1, 2)

        # plot prediction
        if len(pos_pred_indices) > 0:
            ax1.scatter(positive_points[:, 0], positive_points[:, 1], c='g', s=20, label='positive')
        if len(neg_pred_indices) > 0:
            ax1.scatter(negative_points[:, 0], negative_points[:, 1], c='r', s=20, label='negative')

        ax1.set_xlabel('X [m]')
        ax1.set_ylabel('Y [m]')
        ax1.set_title('Prediction')

        # plot confidence scores
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)
        mapper = matplotlib.cm.ScalarMappable(norm=norm)
        mapper.set_array(confidence_scores)
        colors = np.array([mapper.to_rgba(c) for c in confidence_scores])

        ax2.scatter(xyz[:, 0], xyz[:, 1], c=colors, s=10)
        ax2.set_xlabel('X [m]')
        ax2.set_ylabel('Y [m]')
        ax2.set_title('Confidence')
        cbar = plt.colorbar(mapper)
        # cbar.set_label('Confidence', rotation=270, weight='bold')

        plt.legend()
        plt.show()

        # save the predictions (for further sue)
        if predictions_path is not None:
            if not os.path.exists(predictions_path):
                os.makedirs(predictions_path)
            if not os.path.exists(os.path.join(predictions_path, 'true')):
                os.makedirs(os.path.join(predictions_path, 'true'))
            if not os.path.exists(os.path.join(predictions_path, 'false')):
                os.makedirs(os.path.join(predictions_path, 'false'))
            if len(pos_pred_indices) > 0:
                np.save(os.path.join(predictions_path, 'true/poses.npy'), positive_points)
                np.save(os.path.join(predictions_path, 'true/indices.npy'), pos_pred_indices)
            if len(neg_pred_indices) > 0:
                np.save(os.path.join(predictions_path, 'false/poses.npy'), negative_points)
                np.save(os.path.join(predictions_path, 'false/indices.npy'), neg_pred_indices)

    def test(self):
        # load files
        img_paths = load_files(self.img_folder)
        img_kf_paths = load_files(self.img_kf_folder)
        xyz, _ = load_xyz_rot(self.frames_poses_path)
        xyz_kf, _ = load_xyz_rot(self.keyframes_poses_path)
        if self.overlaps_table_path is not None:
            overlaps_table = load_overlaps(self.overlaps_table_path)
        anchors_submaps, _, kf_submaps = load_submaps(self.submaps_folder)

        # load model
        checkpoint = torch.load(self.weights_path)
        self.model.load_state_dict(checkpoint['state_dict'])

        # compute descriptors
        if self.load_descriptors:
            descriptors = np.load(os.path.join(self.descriptors_folder, 'descriptors.npy'))
            descriptors_kf = np.load(os.path.join(self.descriptors_folder, 'descriptors_kf.npy'))
        else:
            print('calculating descriptors for test frames ...')
            descriptors = self.compute_descriptors(img_paths, self.model, anchors_submaps, self.submap_size)

            print('calculating descriptors for keyframe ...')
            descriptors_kf = self.compute_descriptors(img_kf_paths, self.model, kf_submaps, self.submap_size)

            # save descriptors and keyframes descriptors
            if self.descriptors_folder is not None:
                if not os.path.exists(self.descriptors_folder):
                    os.makedirs(self.descriptors_folder)
                np.save(os.path.join(self.descriptors_folder, 'descriptors'), descriptors)
                np.save(os.path.join(self.descriptors_folder, 'descriptors_kf'), descriptors_kf)

        # compute the closest keyframe based on the distances or overlap values (top 1 ground truth)
        if self.overlaps_table_path is not None:
            top_n_keyframes = compute_top_k_keyframes(xyz, xyz_kf, overlaps_table, top_k=1, method=self.method)
        else:
            top_n_keyframes = compute_top_k_keyframes(xyz, xyz_kf, top_k=1, method=self.method)

        # compute the top_n keyframes prediction
        top_n_keyframes_pred = self.compute_predictions(descriptors, descriptors_kf)

        # check per self.skip
        top_n_keyframes_selected = top_n_keyframes[::self.skip]
        top_n_keyframes_pred_selected = top_n_keyframes_pred[::self.skip]
        xyz_selected = xyz[::self.skip]
        xyz_kf_selected = xyz_kf[::self.skip]

        # compute recall and positive and negative prediction indices
        recall, pos_pred_indices, neg_pred_indices, _, _ = self.compute_recall(top_n_keyframes_selected,
                                                                               top_n_keyframes_pred_selected,
                                                                               xyz_selected, xyz_kf_selected)
        print(f'Recall: {recall:.4f}')

        # compute confidence scores
        confidence_scores = self.compute_confidence_scores(descriptors, descriptors_kf, top_n_keyframes_pred)

        # visualize the results
        self.visualization(pos_pred_indices, neg_pred_indices, xyz, xyz_kf, confidence_scores, self.predictions_folder)


if __name__ == '__main__':
    # ===============================================================================
    # loading paths and parameters
    config_filename = '/home/vectr/PycharmProjects/lidar_learning/configs/config.yml'
    params_filename = '/home/vectr/PycharmProjects/lidar_learning/configs/parameters.yml'
    config = yaml.safe_load(open(config_filename))
    params = yaml.safe_load(open(params_filename))

    img_folder = config['data_root']['png_files']                                       # img path for all frames
    overlaps_table_folder = config['data_root']['overlaps']
    poses_folder = config['data_root']['poses']
    keyframes_folder = config['data_root']['keyframes']
    weights_path = config['data_root']['weights']
    descriptors_folder = config['data_root']['descriptors']
    submaps_folder = config['data_root']['submaps']
    submaps_folder = "/media/vectr/vectr3/Dataset/overlap_transformer/submaps/overlap"

    test_seq = config['seqs']['test'][8]
    test_img_folder = os.path.join(img_folder, test_seq)
    test_img_kf_folder = os.path.join(keyframes_folder, test_seq, 'png_files/512')
    test_poses_folder = os.path.join(poses_folder, test_seq, 'poses.txt')
    test_poses_kf_folder = os.path.join(keyframes_folder, test_seq, 'poses/poses_kf.txt')
    weights_path = '/media/vectr/vectr3/Dataset/overlap_transformer/weights/weights_07_15'
    test_weights_path = os.path.join(weights_path, 'best.pth.tar')
    test_overlaps_table_path = os.path.join(overlaps_table_folder, f'{test_seq}.bin')
    test_descriptors_folder = os.path.join(descriptors_folder, test_seq)
    submaps_path = os.path.join(submaps_folder, test_seq)


    # test_img_folder = '/media/vectr/vectr3/Dataset/arl/png_files/out-and-back-3/512'
    # test_img_kf_folder = '/media/vectr/vectr3/Dataset/arl/keyframes/out-and-back-3/png_files/512'
    # test_poses_folder = '/media/vectr/vectr3/Dataset/arl/poses/out-and-back-3/poses.txt'
    # test_poses_kf_folder = '/media/vectr/vectr3/Dataset/arl/keyframes/out-and-back-3/poses/poses_kf.txt'
    # weights_path = '/media/vectr/vectr3/Dataset/overlap_transformer/weights/weights_06_26'
    # test_weights_path = os.path.join(weights_path, 'best.pth.tar')

    # ===============================================================================
    tester = testHandler(params, test_img_folder, test_img_kf_folder, test_poses_folder, test_poses_kf_folder,
                         test_weights_path, overlaps_table_path=test_overlaps_table_path, top_n=5, skip=1, method='euclidean',
                         load_descriptors=False, descriptors_folder=test_descriptors_folder, predictions_folder=None,
                         submaps_path=submaps_path)
    tester.test()

    # tester = testHandler(params, test_img_folder, test_img_kf_folder, test_poses_folder, test_poses_kf_folder,geo:
    # royce:
    # sculpture:
    #                      test_weights_path, overlaps_table_path=None, top_n=5, skip=1, method='euclidean',
    #                      load_descriptors=False, descriptors_folder=None, predictions_folder=None)
    # tester.test()
