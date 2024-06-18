"""
File to compute the submap for each scan. The submap is consisted with the k keyframes.
!!! Current version choose top_k closest keyframes in Euclidean space.
!!! Current version, the submap is only a list of indices of top k keyframes.
"""
import os
import faiss
import numpy as np
import matplotlib.pyplot as plt
from tools.fileloader import load_xyz_rot
from mpl_toolkits import mplot3d


def compute_submap_keyframes(frames_poses_path, keyframes_poses_path, top_k=5):
    # load frames and keyframes positions
    xyz, _ = load_xyz_rot(frames_poses_path)
    xyz_kf, _ = load_xyz_rot(keyframes_poses_path)

    # add keyframes positions to faiss
    index = faiss.IndexFlatL2(3)
    index.add(xyz_kf)

    _, indices = index.search(xyz, top_k)

    visualize_traj_2d(xyz, xyz_kf, indices)







def visualize_traj_3d(xyz, xyz_kf):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    ax.scatter3D(xyz_kf[:, 0], xyz_kf[:, 1], xyz_kf[:, 2], c='gold')
    ax.axis('equal')
    plt.show()


def visualize_traj_2d(xyz, xyz_kf, xyz_kf_selected):
    for i in range(xyz.shape[0]):
        fig, ax = plt.subplots()
        xyz_curr = xyz[i, :]
        xyz_curr_kf = xyz_kf[xyz_kf_selected[i, :], :]

        ax.scatter(xyz[:, 0], xyz[:, 1])
        ax.scatter(xyz_kf[:, 0], xyz_kf[:, 1], c='gold')
        ax.scatter(xyz_curr_kf[:, 0], xyz_curr_kf[:, 1], c='red')
        ax.scatter(xyz_curr[0], xyz_curr[1], c='violet')
        plt.show()
        plt.pause(0.1)
        plt.close(fig)


if __name__ == '__main__':
    seqs = ["bomb_shelter", "botanical_garden", "bruin_plaza", "court_of_sciences", "dickson_court", "geo_loop",
            "kerckhoff", "luskin", "royce_hall", "sculpture_garden"]
    seq = seqs[0]

    root_folder = '/media/vectr/vectr6/Dataset/overlap_transformer'
    frames_poses_path = os.path.join(root_folder, 'poses', seq, 'poses.txt')
    keyframes_poses_path = os.path.join(root_folder, 'keyframes', seq, 'poses', 'poses_kf.txt')

    compute_submap_keyframes(frames_poses_path, keyframes_poses_path)
