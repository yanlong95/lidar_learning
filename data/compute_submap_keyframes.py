"""
File to compute the submap for each scan. The submap is consisted with the k keyframes.
!!! Current version choose top_k closest keyframes in Euclidean space.
!!! Current version, the submap is only a list of indices of top k keyframes.
"""
import os
import faiss
import numpy as np
from tools.fileloader import load_xyz_rot, load_overlaps


def compute_submap_keyframes(frames_poses_path, keyframes_poses_path, overlaps_path, top_k=5, metric='euclidean'):
    """
    Select the top_k keyframes to create a submap for each scan.

    Args:
        frames_poses_path: (string) path of folder contains the positions of each frame
        keyframes_poses_path: (string) path of the folder contains the positions of each keyframes
        overlaps_path: (string) path of the folder contains the overlap table
        top_k: (int) number of top selection.
        metric: (string) selection metric (euclidean or overlap).
    Returns:
        indices: (np.array) the indices of selected top_k keyframes indices in shape (n, top_k).
    """
    # load frames and keyframes positions
    xyz, _ = load_xyz_rot(frames_poses_path)
    xyz_kf, _ = load_xyz_rot(keyframes_poses_path)
    overlaps = load_overlaps(overlaps_path)

    if metric == 'euclidean':
        # search the closest top_k keyframes
        index = faiss.IndexFlatL2(3)
        index.add(xyz_kf)
        _, indices = index.search(xyz, top_k)
    elif metric == 'overlap':
        # search the top_k keyframes base on the overlap values
        index = faiss.IndexFlat(3)
        index.add(xyz)
        _, indices_kf = index.search(xyz_kf, 1)
        overlaps_kf = overlaps[:, indices_kf.squeeze()]             # each row is the overlaps between curr and keyframe

        # search the top_k keyframes for each frame
        indices = np.zeros((len(overlaps), top_k), dtype=int)
        for i in range(len(overlaps)):
            indices[i, :] = overlaps_kf[i, :].argsort()[-top_k:][::-1]
    else:
        raise "Invalid metric! Must be 'euclidean' or 'overlap'"

    return indices


if __name__ == '__main__':
    seqs = ["bomb_shelter", "botanical_garden", "bruin_plaza", "court_of_sciences", "dickson_court", "geo_loop",
            "kerckhoff", "luskin", "royce_hall", "sculpture_garden"]
    seq = seqs[0]

    root_folder = '/Volumes/vectr6/Dataset/overlap_transformer'
    frames_poses_path = os.path.join(root_folder, 'poses', seq, 'poses.txt')
    keyframes_poses_path = os.path.join(root_folder, 'keyframes', seq, 'poses', 'poses_kf.txt')
    overlaps_path = os.path.join(root_folder, 'overlaps', f'{seq}.bin')

    compute_submap_keyframes(frames_poses_path, keyframes_poses_path, overlaps_path, metric='overlap')
