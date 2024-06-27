import os
import numpy as np
import faiss
from tools.fileloader import load_xyz_rot, load_overlaps


def compute_top_k_keyframes(frames_poses, keyframes_poses, overlaps=None, top_k=5, method='euclidean'):
    """
    Select the top_k keyframes for each scan (ground truth).

    Args:
        frames_poses: (numpy.array) positions of all frames in shape (n, 3).
        keyframes_poses: (numpy.array) positions of all keyframes in shape (n, 3).
        overlaps: (numpy.array) overlap table in shape (n, n) element (i, j) is the overlap value between scan i and j.
        top_k: (int) number of top selection.
        method: (string) selection metric (euclidean or overlap).
    Returns:
        indices: (np.array) the indices of selected top_k keyframes indices in shape (n, top_k).
    """
    # # load frames and keyframes positions
    # xyz, _ = load_xyz_rot(frames_poses_path)
    # xyz_kf, _ = load_xyz_rot(keyframes_poses_path)
    # overlaps = load_overlaps(overlaps_path)    # a n*n matrix, element (i, j) is the overlap value between scan i and j

    if method == 'euclidean' or overlaps is None:
        # search the closest top_k keyframes based on the distance in Euclidean space
        index = faiss.IndexFlatL2(3)
        index.add(keyframes_poses)
        _, indices = index.search(frames_poses, top_k)
    elif method == 'overlap':
        # search the top_k keyframes base on the overlap values
        index = faiss.IndexFlat(3)
        index.add(frames_poses)
        _, indices_kf = index.search(keyframes_poses, 1)
        overlaps_kf = overlaps[:, indices_kf.squeeze()]             # each row is the overlaps between curr and keyframe

        # search the top_k keyframes for each frame
        indices = np.zeros((len(overlaps), top_k), dtype=int)
        for i in range(len(overlaps)):
            indices[i, :] = overlaps_kf[i, :].argsort()[-top_k:][::-1]
    else:
        raise "Invalid metric! Must be 'euclidean' or 'overlap'"

    return indices


def compute_top_k_keyframes_prediction(frames_descriptors, keyframes_descriptors, top_k=5, metric='euclidean'):
    pass


def least_squares_points_alignment(src_points, dst_points, weights=None):
    """
    The function to compute translation matrix (T) to minimize the distance between the source points and the
    destination points.

    Args:
        src_points: (np.array) source points in shape (n, 3).
        dst_points: (np.array) destination points in shape (n, 3).
        weights: (np.array) weights corresponding to each point in src_points. If no weights provided, all equal to 1.
    Returns:
        R: (np.array) rotation matrix in shape (3, 3), used for row based points (n, 3), e.g. p' = p @ R.
        t: (np.array) translation vector in shape (3, 3), used for row based points (n, 3), e.g. p' = p + t.
    """
    dim = src_points.shape[1]
    num_src_points = src_points.shape[0]
    num_dst_points = dst_points.shape[0]
    if num_src_points != num_dst_points:
        raise ValueError('Number of points do not match!')

    # input points are n * d (row based), switch to d * n (column based)
    src_points = src_points.T
    dst_points = dst_points.T
    if weights is None:
        weights = np.ones((1, num_src_points))

    p_bar = np.sum(weights * src_points, axis=1, keepdims=True) / np.sum(weights)
    q_bar = np.sum(weights * dst_points, axis=1, keepdims=True) / np.sum(weights)

    X = src_points - p_bar
    Y = dst_points - q_bar
    W = np.diagflat(weights)

    S = X @ W @ Y.T
    U, _, Vh = np.linalg.svd(S)
    U_T = U.T
    V = Vh.T

    M = np.diagflat(np.ones(dim))
    M[-1, -1] = np.linalg.det(V @ U_T)

    R = V @ M @ U_T
    t = q_bar - R @ p_bar

    # return R, t are used for row based points (n * d)
    return R.T, t.T


if __name__ == '__main__':
    seqs = ["bomb_shelter", "botanical_garden", "bruin_plaza", "court_of_sciences", "dickson_court", "geo_loop",
            "kerckhoff", "luskin", "royce_hall", "sculpture_garden"]
    seq = seqs[2]

    root_folder = '/media/vectr/vectr6/Dataset/overlap_transformer'
    frames_poses_path = os.path.join(root_folder, 'poses', seq, 'poses.txt')
    keyframes_poses_path = os.path.join(root_folder, 'keyframes', seq, 'poses', 'poses_kf.txt')
    overlaps_path = os.path.join(root_folder, 'overlaps', f'{seq}.bin')

    xyz, _ = load_xyz_rot(frames_poses_path)
    xyz_kf, _ = load_xyz_rot(keyframes_poses_path)
    overlaps = load_overlaps(overlaps_path)    # a n*n matrix, element (i, j) is the overlap value between scan i and j

    indices = compute_top_k_keyframes(frames_poses_path, keyframes_poses_path, overlaps_path, method='overlap')
