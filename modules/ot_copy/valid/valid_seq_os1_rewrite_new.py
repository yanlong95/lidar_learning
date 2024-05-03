import sys
import tqdm
import faiss
import yaml
import torch
import cv2
from modules.ot_copy.modules.overlap_transformer_haomo import featureExtracter
from modules.ot_copy.tools.utils.utils import *
np.set_printoptions(threshold=sys.maxsize)


def read_image(path):
    depth_data = np.array(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    depth_data_tensor = torch.from_numpy(depth_data).type(torch.FloatTensor).cuda()
    depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)
    depth_data_tensor = torch.unsqueeze(depth_data_tensor, dim=0)

    return depth_data_tensor


def validation(amodel, top_n=5):
    # ===============================================================================
    # loading paths and parameters
    config_filename = '/home/vectr/PycharmProjects/lidar_learning/configs/config.yml'
    config = yaml.safe_load(open(config_filename))
    valid_scans_folder = config['data_root']['png_files']
    ground_truth_folder = config['data_root']['gt_overlaps']
    valid_seq = config['seqs']['valid'][0]
    # ===============================================================================

    valid_scan_paths = load_files(os.path.join(valid_scans_folder, '900', valid_seq))
    ground_truth_paths = os.path.join(ground_truth_folder, valid_seq, 'overlaps.npz')
    ground_truth_overlaps = np.load(ground_truth_paths)['arr_0']

    with torch.no_grad():
        num_scans = len(valid_scan_paths)

        # calculate all descriptors
        descriptors = np.zeros((num_scans, 256))
        for i in tqdm.tqdm(range(num_scans)):
            # load a scan
            current_batch = read_image(valid_scan_paths[i])
            current_batch = torch.cat((current_batch, current_batch), dim=0)        # no idea why, keep it now

            # calculate descriptor
            amodel.eval()
            current_descriptor = amodel(current_batch)
            descriptors[i, :] = current_descriptor[0, :].cpu().detach().numpy()

        descriptors = descriptors.astype('float32')

        # searching
        d = descriptors.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(descriptors)

        # search the closest descriptors for each one in the validation set
        """
        2 ways to calculate the validation.
        a. search the closest descriptors, if their overlap values greater than the threshold, then positive prediction
           (not recommend as most of top_n scans are close to current scan even for a random model. choose a large top_n
           if you want to use this way).
        b. search top_n positive and negative descriptors, if distances between the all positive descriptors are smaller 
           than the negative descriptors, then positive prediction.  
        """
        # TODO: add another valid method. choose the top n closest keyframes based on the distance of descriptors, if
        #       any chosen keyframe is the closest keyframe based on global distance, then a correct prediction.

        recall = True
        num_pos_pred = 0

        for i in range(num_scans):
            ground_truth = ground_truth_overlaps[i]
            pos_scans = ground_truth[ground_truth[:, 2] >= ground_truth[:, 3]]
            neg_scans = ground_truth[ground_truth[:, 2] < ground_truth[:, 3]]

            # in case no enough positive scans
            top_n = min(len(pos_scans), top_n)

            if recall:
                D, I = index.search(descriptors[i, :].reshape(1, -1), top_n)

                if I[:, 0] == i:
                    min_index = I[:, 1:]
                else:
                    min_index = I[:, :]

                if np.all(ground_truth[min_index, 2] > ground_truth[min_index, 3]):
                    num_pos_pred += 1
            else:
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

    # precision = num_pos_pred / (top_n * num_valid)
    precision = num_pos_pred / num_scans
    # precision_neg = num_neg_pred / num_valid
    print(f'top {top_n} precision: {precision}.')
    # print(f'top {top_n} precision_neg: {precision_neg}')
    return precision


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amodel = featureExtracter(height=32, width=900, channels=1, use_transformer=True).to(device)
    validation(amodel)
