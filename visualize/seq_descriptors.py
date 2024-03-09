# temp file to test sequential descriptors

import os
import yaml
import numpy as np
import faiss

from tools.fileloader import load_xyz_rot, load_descriptors


# concatenate n descriptors together to create a sequential descriptor
def stack_descriptors(descriptors, keyframe_indices, n=5):
    num, length = descriptors.shape
    descriptors_stack = np.zeros((num + n - 1, length * n))
    for i in range(n):
        descriptors_stack[i:i+num, (n-i-1)*length:(n-i)*length] = descriptors

    # sequential descriptor stack with shape (num-n+1, length*n), each row is a new descriptor in (t, t-1, t-2 ...)
    descriptors_seq = descriptors_stack[n-1:num, :]

    # we ignore the first keyframe since it is always index 0 (cannot find in the sequence).
    indices = keyframe_indices[1:]
    indices = indices - n + 1
    descriptors_kf_seq = descriptors_seq[indices, :]

    return descriptors_seq, descriptors_kf_seq


# find the indices of keyframes
def find_keyframe_indices(poses, poses_kf):
    index_poses = faiss.IndexFlatL2(xyz.shape[1])
    index_poses.add(poses)
    distances, ids = index_poses.search(poses_kf, 1)
    keyframe_indies = np.squeeze(ids)
    return keyframe_indies


if __name__ == '__main__':
    config_path = '../configs/plot_temp.yaml'
    config = yaml.safe_load(open(config_path))

    seq = 'sculpture_garden'
    poses_folder_path = config['data_root']['poses']
    descriptors_folder_path = config['data_root']['descriptors']
    keyframes_folder_path = config['data_root']['keyframes']

    poses_path = os.path.join(poses_folder_path, seq, 'poses.txt')
    keyframe_poses_path = os.path.join(keyframes_folder_path, seq, 'poses/poses_kf.txt')

    descriptors_path = os.path.join(descriptors_folder_path, seq, 'test_whole_frame_descriptors.npy')
    keyframe_descriptors_path = os.path.join(descriptors_folder_path, seq, 'keyframe_descriptors.npy')

    xyz, _ = load_xyz_rot(poses_path)
    xyz_kf, _ = load_xyz_rot(keyframe_poses_path)

    descriptors = load_descriptors(descriptors_path)
    descriptors_kf = load_descriptors(keyframe_descriptors_path)

    keyframe_indices = find_keyframe_indices(xyz, xyz_kf)
    descriptors_seq, descriptors_kf_seq = stack_descriptors(descriptors, keyframe_indices)

    test_selection = 10
    # select 1 sample per test_selection samples, reduce the test size
    test_frame_descriptors = descriptors_seq[::test_selection]

    np.save(os.path.join(descriptors_folder_path, seq, 'keyframe_seq_descriptors'), descriptors_kf_seq)
    np.save(os.path.join(descriptors_folder_path, seq, 'test_frame_seq_descriptors'), test_frame_descriptors)
