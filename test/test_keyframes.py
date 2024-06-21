# p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
# if p not in sys.path:
#     sys.path.append(p)
import os
import sys
from matplotlib import pyplot as plt
import matplotlib
import torch
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay, transform
from modules.overlap_transformer import OverlapTransformer32
from tools.fileloader import load_files, load_xyz_rot, read_image
from tools.utils_func import compute_top_k_keyframes
import faiss
import yaml
import tqdm


def load_keyframes(keyframe_img_path, keyframe_poses_path):
    keyframe_images = load_files(keyframe_img_path)
    keyframe_xyz, _ = load_xyz_rot(keyframe_poses_path)

    return keyframe_images, keyframe_xyz


def load_test_frames(test_frame_img_path, test_frame_poses_path):
    test_frame_images = load_files(test_frame_img_path)
    test_frame_xyz, _ = load_xyz_rot(test_frame_poses_path)

    return test_frame_images, test_frame_xyz


# load test frame ground truth overlaps for debugging
def load_test_frame_overlaps(test_frames_path):
    test_frame_overlaps_folder = os.path.join(test_frames_path, 'overlaps_test')
    test_frame_overlaps_path = load_files(test_frame_overlaps_folder)
    test_frame_overlaps = []
    for i in range(len(test_frame_overlaps_path)):
        test_frame_overlap = np.load(test_frame_overlaps_path[i])
        test_frame_overlaps.append(test_frame_overlap)

    return test_frame_overlaps


# The ground truth based on the location of the keyframes (e.g. choose the closest top n keyframes based on the
# location).
def calc_ground_truth(poses):
    frame_loc = poses[:, :3, 3]
    frame_qua = np.zeros((len(poses), 4))

    for i in range(len(poses)):
        rotation_matrix = poses[i, :3, :3]
        rotation = transform.Rotation.from_matrix(rotation_matrix)
        quaternion = rotation.as_quat()
        frame_qua[i, :] = quaternion

    return frame_loc, frame_qua


def calc_descriptors(images, amodel):
    n_keyframes = len(images)
    descriptors = np.zeros((n_keyframes, 256))

    with torch.no_grad():
        for i in tqdm.tqdm(range(n_keyframes)):
            curr_batch = read_image(images[i])
            curr_batch = torch.cat((curr_batch, curr_batch), dim=0)

            amodel.eval()
            curr_descriptor = amodel(curr_batch)
            descriptors[i, :] = curr_descriptor[0, :].cpu().detach().numpy()

    descriptors = descriptors.astype('float32')
    return descriptors


def calc_voronoi_map(keyframe_poses):
    xy = keyframe_poses[:, :2]
    voronoi = Voronoi(xy, incremental=True)

    # voronoi map is unbounded, set boundary by adding corner points
    x_min = min(xy[:, 0])
    x_max = max(xy[:, 0])
    y_min = min(xy[:, 1])
    y_max = max(xy[:, 1])
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    x_size = (x_max - x_min) / 2
    y_size = (y_max - y_min) / 2

    lower_left = [x_center - 1.2 * x_size, y_center - 1.2 * y_size]
    lower_right = [x_center + 1.2 * x_size, y_center - 1.2 * y_size]
    upper_left = [x_center - 1.2 * x_size, y_center + 1.2 * y_size]
    upper_right = [x_center + 1.2 * x_size, y_center + 1.2 * y_size]
    boundary = np.array([lower_left, lower_right, upper_left, upper_right])
    voronoi.add_points(boundary)

    # center for each region
    voronoi_centers = voronoi.points[:-len(boundary)]

    # extract the vertices positions for each region
    vertices = voronoi.vertices                 # vertices positions
    regions = voronoi.regions                   # indices of vertices for each region
    point_region = voronoi.point_region         # voronoi region (index of polyhedron)

    voronoi_vertices_indices = [regions[point_region[i]] for i in range(len(point_region) - len(boundary))]
    voronoi_vertices_poses = []                 # the vertices for all regions
    delaunay_triangulation = []                 # create delaunay triangulation to determine if a point is inside region

    for voronoi_vertex_index in voronoi_vertices_indices:
        vertices_poses = [vertices[idx] for idx in voronoi_vertex_index if idx > -1]
        voronoi_vertices_poses.append(vertices_poses)
        delaunay_triangulation.append(Delaunay(vertices_poses))

    # double check if the point lies within the region for all the regions of the map
    for i in range(len(voronoi_vertices_poses)):
        point = voronoi_centers[i]
        vertex = np.array(voronoi_vertices_poses[i])
        tri = delaunay_triangulation[i]

        if tri.find_simplex(point) < 0:                     # True if the point lies outside the region
            print(f'point {i} lies outside the region.')

        # fig = voronoi_plot_2d(voronoi)
        # plt.scatter(point[0], point[1], c='b', s=50, label='center')
        # plt.scatter(vertex[:, 0], vertex[:, 1], c='r', s=100, label='vertices')
        # plt.legend()
        # plt.show()

    # fig = voronoi_plot_2d(voronoi)
    # plt.show()

    return delaunay_triangulation, voronoi


def calc_top_n_voronoi(keyframe_poses, keyframe_descriptors, keyframe_voronoi_region, test_frame_poses,
                       test_frame_descriptors, top_n=5):
    num_test_frames = len(test_frame_poses)

    # initialize searching
    nlist = 1
    dim_pose = 3
    dim_descriptor = 256

    quantizer_poses = faiss.IndexFlatL2(dim_pose)
    quantizer_descriptors = faiss.IndexFlatL2(dim_descriptor)

    index_poses = faiss.IndexIVFFlat(quantizer_poses, dim_pose, nlist, faiss.METRIC_L2)
    index_descriptors = faiss.IndexIVFFlat(quantizer_descriptors, dim_descriptor, nlist, faiss.METRIC_L2)
    # check if descriptors are normalized !!!!!!!!!
    # index_descriptors = faiss.IndexIVFFlat(quantizer_descriptors, dim_descriptor, nlist, faiss.METRIC_INNER_PRODUCT)

    if not index_poses.is_trained:
        index_poses.train(keyframe_poses)

    if not index_descriptors.is_trained:
        index_descriptors.train(keyframe_descriptors)

    index_poses.add(keyframe_poses)
    index_descriptors.add(keyframe_descriptors)

    positive_pred = []
    negative_pred = []
    top_n_choices = []
    positive_pred_indices = []
    negative_pred_indices = []

    for curr_frame_idx in range(num_test_frames):
        curr_frame_pose = test_frame_poses[curr_frame_idx, :].reshape(1, -1)                        # (dim,) to (1, dim)
        curr_frame_descriptor = test_frame_descriptors[curr_frame_idx, :].reshape(1, -1)

        # searching top n poses and descriptors
        # D_pose, I_pose = index_poses.search(curr_frame_pose, top_n)
        D_descriptor, I_descriptor = index_descriptors.search(curr_frame_descriptor, top_n)

        # determine if a point inside the regions
        top_n_keyframes_indices = I_descriptor[0]
        top_n_keyframes_regions = [keyframe_voronoi_region[idx] for idx in top_n_keyframes_indices]
        top_n_choices.append(top_n_keyframes_indices)

        for idx in range(top_n):
            pos_2d = curr_frame_pose[0][:2]
            pos_3d = curr_frame_pose[0][:3]
            region = top_n_keyframes_regions[idx]

            if region.find_simplex(pos_2d) >= 0:  # True if the point lies inside the region
                positive_pred.append(pos_3d)
                positive_pred_indices.append(curr_frame_idx)
                break

            if idx == top_n - 1:
                negative_pred.append(pos_3d)
                negative_pred_indices.append(curr_frame_idx)

    precision = len(positive_pred) / num_test_frames
    print(f'Prediction precision: {precision}.')

    return precision, positive_pred, negative_pred, positive_pred_indices, negative_pred_indices, top_n_choices


def calc_top_n_distance(keyframe_poses, keyframe_descriptors, test_frame_poses, test_frame_descriptors, top_n, max_dist):
    positive_pred = []
    negative_pred = []
    top_n_choices = []
    positive_pred_indices = []
    negative_pred_indices = []

    num_test_frames = test_frame_descriptors.shape[0]
    dim_descriptors = keyframe_descriptors.shape[1]

    # search the min distances between any 2 keyframes and use it as the threshold for positive and negative prediction
    index_kf_poses = faiss.IndexFlatL2(3)
    index_kf_poses.add(keyframe_poses)
    D_kf_poses, _ = index_kf_poses.search(keyframe_poses, 2)
    max_dist = max(max(D_kf_poses[:, 1]) ** 0.5, max_dist)

    # faiss search based on descriptors norm distances
    index_descriptors = faiss.IndexFlatL2(dim_descriptors)
    # index_descriptors = faiss.IndexFlatL2(dim_descriptors, faiss.METRIC_INNER_PRODUCT)
    index_descriptors.add(keyframe_descriptors)

    # check prediction for each test frame
    D_descriptors, I_descriptors = index_descriptors.search(test_frame_descriptors, top_n)
    for i in range(num_test_frames):
        curr_frame_poses = test_frame_poses[i, :]
        top_n_keyframes_indices = I_descriptors[i, :]
        top_n_keyframes_distances = np.linalg.norm(keyframe_poses[top_n_keyframes_indices, :] - curr_frame_poses,
                                                   axis=1)

        # check if positive or negative prediction
        top_n_choices.append(top_n_keyframes_indices)
        if np.any(top_n_keyframes_distances <= max_dist):
            positive_pred.append(curr_frame_poses)
            positive_pred_indices.append(i)
        else:
            negative_pred.append(curr_frame_poses)
            negative_pred_indices.append(i)

    precision = len(positive_pred) / num_test_frames
    print(f'Prediction precision: {precision}.')

    return precision, positive_pred, negative_pred, positive_pred_indices, negative_pred_indices, top_n_choices


def calc_top_n_confidence_score(positive_pred_indices, negative_pred_indices, descriptors, descriptors_kf,
                                top_n_choices, use_min=True):
    num_frames = descriptors.shape[0]
    confidence_scores = np.zeros(num_frames)
    for idx in positive_pred_indices:
        curr_descriptor = descriptors[idx, :]
        top_n_choice = top_n_choices[idx]
        top_n_descriptors = descriptors_kf[top_n_choice, :]
        if use_min:
            top_n_best_dist = np.min(np.linalg.norm(curr_descriptor - top_n_descriptors, axis=1))
        else:
            top_n_best_dist = np.mean(np.linalg.norm(curr_descriptor - top_n_descriptors, axis=1))
        confidence_scores[idx] = top_n_best_dist

    for idx in negative_pred_indices:
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


def plot_prediction(positive_pred, negative_pred, positive_pred_indices, negative_pred_indices,
                    predictions_path, confidence_scores, test_frame_poses):
    # fig = voronoi_plot_2d(voronoi_map)
    positive_points = np.array(positive_pred)
    negative_points = np.array(negative_pred)

    # save the predictions (for further use)
    if not os.path.exists(predictions_path):
        os.makedirs(predictions_path)
    if not os.path.exists(os.path.join(predictions_path, 'true')):
        os.makedirs(os.path.join(predictions_path, 'true'))
    if not os.path.exists(os.path.join(predictions_path, 'false')):
        os.makedirs(os.path.join(predictions_path, 'false'))
    np.save(os.path.join(predictions_path, 'true/poses.npy'), positive_points)
    np.save(os.path.join(predictions_path, 'false/poses.npy'), negative_points)
    np.save(os.path.join(predictions_path, 'true/indices.npy'), positive_pred_indices)
    np.save(os.path.join(predictions_path, 'false/indices.npy'), negative_pred_indices)

    fig, [ax1, ax2] = plt.subplots(1, 2)

    # plot prediction
    if len(positive_pred) > 0:
        ax1.scatter(positive_points[:, 0], positive_points[:, 1], c='g', s=50, label='positive')
    if len(negative_pred) > 0:
        ax1.scatter(negative_points[:, 0], negative_points[:, 1], c='r', s=50, label='negative')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_title('Prediction')

    # plot confidence
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)
    mapper = matplotlib.cm.ScalarMappable(norm=norm)
    mapper.set_array(confidence_scores)
    colors = np.array([mapper.to_rgba(c) for c in confidence_scores])

    ax2.scatter(test_frame_poses[:, 0], test_frame_poses[:, 1], c=colors, s=10)
    ax2.set_xlabel('X [m]')
    ax2.set_ylabel('Y [m]')
    ax2.set_title('Confidence')
    cbar = plt.colorbar(mapper)
    # cbar.set_label('Confidence', rotation=270, weight='bold')

    plt.legend()
    plt.show()

    # plt.scatter(vertex[:, 0], vertex[:, 1], c='r', s=100, label='vertices')


def plot_top_n_keyframes(positive_pred_indices, negative_pred_indices, top_n_choices, keyframe_poses, test_frame_poses,
                         test_frame_poses_full):
    for idx in negative_pred_indices:
        # load the top n keyframes poses for current frame
        top_n_choice = top_n_choices[idx]
        top_n_keyframe_poses = np.array([keyframe_poses[i][:2] for i in top_n_choice])

        # load ground truth overlap for current frame
        # test_frame_overlap = test_frame_overlaps[idx][:, 2]
        # test_frame_indices = np.argsort(test_frame_overlap)
        # test_frame_poses_sorted = test_frame_poses_full[test_frame_indices]

        # plot
        plt.figure(1)
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)
        mapper = matplotlib.cm.ScalarMappable(norm=norm)
        # mapper.set_array(test_frame_overlap)
        # colors = np.array([mapper.to_rgba(1) if a > 0.3 else mapper.to_rgba(0) for a in test_frame_overlap])

        # plt.scatter(test_frame_poses_sorted[:, 0], test_frame_poses_sorted[:, 1], c=colors[test_frame_indices], s=10)
        plt.scatter(keyframe_poses[:, 0], keyframe_poses[:, 1], c='tan', s=5, label='keyframes')
        plt.scatter(top_n_keyframe_poses[:, 0], top_n_keyframe_poses[:, 1], c='magenta', s=5, label='top n choices')
        plt.scatter(test_frame_poses[idx, 0], test_frame_poses[idx, 1], c='orange', s=20, label=f'frame: {idx}')

        # plt.axis('square')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.title('Overlap Map')
        plt.legend()
        # cbar = plt.colorbar(mapper)
        # cbar.set_label('Overlap', rotation=270, weight='bold')

        plt.show()





# def plot_top_n_confidence_score(positive_pred_indices, negative_pred_indices, descriptors, descriptors_kf,
#                                 top_n_choices, poses, use_min=True):
#     num_frames = descriptors.shape[0]
#     confidence_scores = np.zeros(num_frames)
#     for idx in positive_pred_indices:
#         curr_descriptor = descriptors[idx, :]
#         top_n_choice = top_n_choices[idx]
#         top_n_descriptors = descriptors_kf[top_n_choice, :]
#         if use_min:
#             top_n_best_dist = np.min(np.linalg.norm(curr_descriptor - top_n_descriptors, axis=1))
#         else:
#             top_n_best_dist = np.mean(np.linalg.norm(curr_descriptor - top_n_descriptors, axis=1))
#         confidence_scores[idx] = top_n_best_dist
#
#     for idx in negative_pred_indices:
#         curr_descriptor = descriptors[idx, :]
#         top_n_choice = top_n_choices[idx]
#         top_n_descriptors = descriptors[top_n_choice, :]
#         if use_min:
#             top_n_best_dist = np.min(np.linalg.norm(curr_descriptor - top_n_descriptors, axis=1))
#         else:
#             top_n_best_dist = np.mean(np.linalg.norm(curr_descriptor - top_n_descriptors, axis=1))
#         confidence_scores[idx] = top_n_best_dist
#
#     confidence_scores /= np.max(confidence_scores)
#     confidence_scores = np.ones_like(confidence_scores) - confidence_scores
#
#     plt.figure(2)
#     norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)
#     mapper = matplotlib.cm.ScalarMappable(norm=norm)
#     mapper.set_array(confidence_scores)
#     colors = np.array([mapper.to_rgba(c) for c in confidence_scores])
#
#     plt.scatter(poses[:, 0], poses[:, 1], c=colors, s=10)
#
#     plt.xlabel('X [m]')
#     plt.ylabel('Y [m]')
#     plt.title('Confidence Map')
#     plt.legend()
#     cbar = plt.colorbar(mapper)
#     cbar.set_label('Confidence', rotation=270, weight='bold')
#     plt.show()


def testHandler(test_frame_img_path, test_frame_poses_path, keyframes_img_path, keyframes_poses_path, weights_path,
                descriptors_path, predictions_path,  test_selection=1, load_descriptors=False,
                metric='euclidean'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amodel = OverlapTransformer32(height=32, width=512, channels=1, use_transformer=True).to(device)

    with ((torch.no_grad())):
        # load model
        print(f'Load weights from {weights_path}')
        checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
        amodel.load_state_dict(checkpoint['state_dict'])

        # calculate ground truth and descriptors
        test_frame_images, test_frame_xyz = load_test_frames(test_frame_img_path, test_frame_poses_path)
        keyframe_images, keyframe_xyz = load_keyframes(keyframes_img_path, keyframes_poses_path)
        # test_frame_overlaps_full = load_test_frame_overlaps(test_frames_path)

        test_frame_xyz_selected = test_frame_xyz[::test_selection]
        # test_frame_overlaps = test_frame_overlaps_full[::test_selection]

        if load_descriptors:
            keyframe_descriptors = np.load(os.path.join(descriptors_path, 'keyframe_descriptors.npy'))
            test_frame_descriptors = np.load(os.path.join(descriptors_path, 'test_frame_descriptors.npy'))
            test_frame_descriptors_selected = test_frame_descriptors[::test_selection]
        else:
            print('calculating descriptors for keyframe ...')
            keyframe_descriptors = calc_descriptors(keyframe_images, amodel)

            print('calculating descriptors for test frames ...')
            test_frame_descriptors = calc_descriptors(test_frame_images, amodel)

            # select 1 sample per test_selection samples, reduce the test size
            test_frame_descriptors_selected = test_frame_descriptors[::test_selection]

            if not os.path.exists(descriptors_path):
                os.makedirs(descriptors_path)
            np.save(os.path.join(descriptors_path, 'keyframe_descriptors'), keyframe_descriptors)
            np.save(os.path.join(descriptors_path, 'test_frame_descriptors'), test_frame_descriptors)

        # # test 2d plot
        # plot_keyframe_poses(test_frame_xyz, keyframe_xyz)

        if metric == 'euclidean':
            # max_dist = max(max_dist, maximum distances between any 2 keyframes)
            precision, positive_pred, negative_pred, positive_pred_indices, negative_pred_indices, top_n_choices = \
                calc_top_n_distance(keyframe_xyz, keyframe_descriptors, test_frame_xyz_selected,
                                    test_frame_descriptors_selected, top_n=5, max_dist=5)
        elif metric == 'voronoi':

            # test voronoi map
            keyframe_voronoi_region, voronoi_map = calc_voronoi_map(keyframe_xyz)
            precision, positive_pred, negative_pred, positive_pred_indices, negative_pred_indices, top_n_choices = \
                calc_top_n_voronoi(keyframe_xyz, keyframe_descriptors, keyframe_voronoi_region, test_frame_xyz_selected,
                                   test_frame_descriptors_selected, top_n=5)
        else:
            raise ValueError('Invalid metric! Metric must be either euclidean or voronoi!')

        # show the result
        confidence_scores = calc_top_n_confidence_score(positive_pred_indices, negative_pred_indices,
                                                        test_frame_descriptors, keyframe_descriptors, top_n_choices,
                                                        use_min=True)
        plot_prediction(positive_pred, negative_pred, positive_pred_indices, negative_pred_indices, predictions_path,
                        confidence_scores, test_frame_xyz)
        # plot_top_n_keyframes(positive_pred_indices, negative_pred_indices, top_n_choices, keyframe_xyz,
        #                      test_frame_xyz_selected, test_frame_xyz)
        # plot_top_n_confidence_score(positive_pred_indices, negative_pred_indices, test_frame_descriptors,
        #                             keyframe_descriptors, top_n_choices, test_frame_xyz, use_min=True)


if __name__ == '__main__':
    # load config ================================================================
    config_filename = '/home/vectr/PycharmProjects/lidar_learning/configs/config.yml'
    config = yaml.safe_load(open(config_filename))
    frame_img_path = config["data_root"]["png_files"]
    frame_poses_path = config["data_root"]["poses"]
    keyframe_path = config["data_root"]["keyframes"]
    descriptors_path = config["data_root"]["descriptors"]
    predictions_path = config["data_root"]["predictions"]
    overlaps_path = config["data_root"]["overlaps"]
    weights_path = config["data_root"]["weights"]
    test_seq = config["seqs"]["test"][9]
    # ============================================================================

    test_frame_img_path = os.path.join(frame_img_path, '512', test_seq)
    test_frame_pose_path = os.path.join(frame_poses_path, test_seq, 'poses.txt')
    test_keyframe_img_path = os.path.join(keyframe_path, test_seq, 'png_files', '512')
    test_keyframe_poses_path = os.path.join(keyframe_path, test_seq, 'poses', 'poses_kf.txt')
    test_weights_path = os.path.join(weights_path, 'last.pth.tar')
    test_descriptors_path = os.path.join(descriptors_path, test_seq)
    test_predictions_path = os.path.join(predictions_path, test_seq)
    test_weights_path = '/media/vectr/vectr3/Dataset/overlap_transformer/weights/weights_06_13/last.pth.tar'

    # seqs = ['mout-forest-loop', 'mout-loop-1', 'mout-water-2', 'mout-water-3_filtered', 'mout-with-vehicles',
    #         'out-and-back-2', 'out-and-back-3', 'parking-lot']
    # seq = seqs[5]

    # test_frame_img_path = os.path.join(f'/media/vectr/vectr3/Dataset/arl/png_files/{seq}', '512')
    # test_frame_pose_path = os.path.join(f'/media/vectr/vectr3/Dataset/arl/poses/{seq}', 'poses.txt')
    # test_keyframe_img_path = os.path.join(f'/media/vectr/vectr3/Dataset/arl/keyframes/{seq}', 'png_files', '512')
    # test_keyframe_poses_path = os.path.join(f'/media/vectr/vectr3/Dataset/arl/poses/{seq}', 'poses_kf.txt')
    # test_weights_path = os.path.join('/media/vectr/vectr3/Dataset/overlap_transformer/weights/weights_512_schedule', 'last.pth.tar')
    # test_descriptors_path = os.path.join('/media/vectr/vectr3/Dataset/arl/descriptors', seq)
    # test_predictions_path = os.path.join('/media/vectr/vectr3/Dataset/arl/predictions', seq)

    # test_frame_img_path = os.path.join(f'/media/vectr/vectr3/Dataset/loop_closure_detection/png_files/e4_3', '512')
    # test_frame_pose_path = os.path.join(f'/media/vectr/vectr3/Dataset/loop_closure_detection/poses/e4_3', 'poses.txt')
    # test_keyframe_img_path = os.path.join(f'/media/vectr/vectr3/Dataset/loop_closure_detection/keyframes/e4_3', 'png_files', '512')
    # test_keyframe_poses_path = os.path.join(f'/media/vectr/vectr3/Dataset/loop_closure_detection/poses/e4_3', 'poses_kf.txt')
    # test_weights_path = os.path.join('/media/vectr/vectr3/Dataset/overlap_transformer/weights/weights_512_schedule',
    #                                  'last.pth.tar')
    # test_descriptors_path = os.path.join('/media/vectr/vectr3/Dataset/loop_closure_detection/descriptors', 'e4_3')
    # test_predictions_path = os.path.join('/media/vectr/vectr3/Dataset/loop_closure_detection/predictions', 'e4_3')

    testHandler(test_frame_img_path, test_frame_pose_path, test_keyframe_img_path, test_keyframe_poses_path,
                test_weights_path, test_descriptors_path, test_predictions_path, test_selection=1,
                load_descriptors=True, metric='voronoi')
