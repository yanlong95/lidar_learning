"""
File to compute the submap for each scan. The submap is consisted with the k keyframes.
!!! Current version choose top_k closest keyframes in Euclidean space.
!!! Current version, the submap is only a list of indices of top k keyframes.
!!! Current version, euclidean distance is more stable than overlap.
"""
import os
import faiss
import numpy as np
import matplotlib.pyplot as plt
from tools.fileloader import load_xyz_rot, load_overlaps


def compute_keyframes_indices(xyz, xyz_kf):
    """
    Compute the indices of keyframes.

    Args:
        xyz: (numpy.array) frames positions in shape (n, 3).
        xyz_kf: (numpy.array) keyframes positions in shape (m, 3).
    Returns:
        indices_kf: (np.array) the indices of keyframes in shape (n).
    """
    index = faiss.IndexFlatL2(3)
    index.add(xyz)
    _, indices_kf = index.search(xyz_kf, 1)
    indices_kf = indices_kf.squeeze()
    return indices_kf


def compute_submap_keyframes(xyz, xyz_kf, overlaps, top_k=5, is_anchor=False, overlap_dist_thresh=15.0,
                             metric='euclidean'):
    """
    Select the top_k keyframes to create a submap for each scan.

    Args:
        xyz: (numpy.array) frames positions in shape (n, 3).
        xyz_kf: (numpy.array) keyframes positions in shape (m, 3).
        overlaps_path: (string) path of the folder contains the overlap table
        top_k: (int) number of top selection.
        is_anchor: (bool) whether to compute the submap for anchor or not. Anchor does not know the keyframes after
        current time step, the submap for anchor only considers the previous keyframes.
        metric: (string) selection metric (euclidean or overlap).
    Returns:
        indices: (np.array) the indices of selected top_k keyframes indices in shape (n, top_k).
    """
    if metric == 'euclidean':
        # search the closest top_k keyframes
        index_kf = faiss.IndexFlatL2(3)
        index_kf.add(xyz_kf)
        if not is_anchor:
            _, indices = index_kf.search(xyz, top_k)
        else:
            # cannot choose keyframes before current index
            indices_kf = compute_keyframes_indices(xyz, xyz_kf)
            _, indices_all = index_kf.search(xyz, len(xyz_kf))            # rank keyframes for each frame (in kf order)
            indices = np.zeros((len(overlaps), top_k), dtype=int)
            for i in range(len(xyz)):
                rank_kf = indices_all[i, :]          # mask in keyframe order (0, 1, 2, ...)
                mask = indices_kf[rank_kf] <= i      # mask in frame order (0, 111, 187, ...)

                if np.sum(mask) >= top_k:
                    indices[i, :] = indices_all[i, mask][:top_k]
                else:
                    indices[i, :np.sum(mask)] = indices_all[i, mask]

    elif metric == 'overlap':
        # search the top_k keyframes base on the overlap values
        indices_kf = compute_keyframes_indices(xyz, xyz_kf)
        overlaps_kf = overlaps[:, indices_kf]             # each row is the overlaps between curr and keyframe

        # search the top_k keyframes for each frame
        indices = np.zeros((len(overlaps), top_k), dtype=int)

        for i in range(len(overlaps)):
            rank_kf = overlaps_kf[i, :].argsort()[::-1]
            xyz_curr = xyz[i, :]
            if not is_anchor:
                # add minimum distance threshold
                valid_count = 0
                for j in range(len(xyz_kf)):
                    xyz_kf_curr = xyz_kf[rank_kf[j], :]
                    if np.linalg.norm(xyz_curr - xyz_kf_curr) <= overlap_dist_thresh:
                        indices[i, valid_count] = rank_kf[j]
                        valid_count += 1
                    if valid_count >= top_k:
                        break
                if valid_count < top_k:
                    raise "Overlap distance threshold is too small! Cannot find enough keyframes."
                # indices[i, :] = rank_kf[:top_k]
            else:
                # anchor cannot choose keyframes before current index
                mask = indices_kf[rank_kf] <= i
                if np.sum(mask) >= top_k:
                    # add minimum distance threshold
                    valid_count = 0
                    for j in range(len(xyz_kf)):
                        xyz_kf_curr = xyz_kf[rank_kf[j], :]
                        if np.linalg.norm(xyz_curr - xyz_kf_curr) <= overlap_dist_thresh:
                            indices[i, valid_count] = rank_kf[j]
                            valid_count += 1
                        if valid_count >= top_k:
                            break
                    if valid_count < top_k:
                        raise "Overlap distance threshold is too small! Cannot find enough keyframes."
                    # indices[i, :] = rank_kf[:top_k]
                else:
                    indices[i, :np.sum(mask)] = rank_kf[mask]
    else:
        raise "Invalid metric! Must be 'euclidean' or 'overlap'"

    return indices


if __name__ == '__main__':
    seqs = ["bomb_shelter", "botanical_garden", "bruin_plaza", "court_of_sciences", "dickson_court", "geo_loop",
            "kerckhoff", "luskin", "royce_hall", "sculpture_garden"]
    seq = seqs[2]

    root_folder = '/media/vectr/vectr3/Dataset/overlap_transformer'
    frames_poses_path = os.path.join(root_folder, 'poses', seq, 'poses.txt')
    keyframes_poses_path = os.path.join(root_folder, 'keyframes', seq, 'poses', 'poses_kf.txt')
    overlaps_path = os.path.join(root_folder, 'overlaps', f'{seq}.bin')

    xyz, _ = load_xyz_rot(frames_poses_path)
    xyz_kf, _ = load_xyz_rot(keyframes_poses_path)
    overlaps = load_overlaps(overlaps_path)
    indices_kf = compute_keyframes_indices(xyz, xyz_kf)
    indices = compute_submap_keyframes(xyz, xyz_kf, overlaps, is_anchor=False, overlap_dist_thresh=15.0,
                                       metric='overlap')

    # for i in range(xyz.shape[0]):
    #     xyz_curr = xyz[i, :]
    #     xyz_curr_kf = xyz_kf[indices[i, :], :]
    #
    #     plt.clf()
    #     plt.scatter(xyz[:, 0], xyz[:, 1])
    #     plt.scatter(xyz_kf[:, 0], xyz_kf[:, 1], c='gold')
    #     plt.scatter(xyz_curr_kf[:, 0], xyz_curr_kf[:, 1], c='red')
    #     plt.scatter(xyz_curr[0], xyz_curr[1], c='violet')
    #     plt.show()
    #     plt.pause(0.01)
