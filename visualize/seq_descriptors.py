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

    # we ignore the first keyframe since it is always index 0 (cannot find in the sequence), instead we use the n_th
    # frame as the first keyframe

    # print(keyframe_indices)
    keyframe_indices = keyframe_indices - n + 1
    keyframe_indices[keyframe_indices < 0] = 0
    # print(keyframe_indices)
    descriptors_kf_seq = descriptors_seq[keyframe_indices, :]

    return descriptors_seq, descriptors_kf_seq


# find the indices of keyframes
def find_keyframe_indices(poses, poses_kf, n=5):
    index_poses = faiss.IndexFlatL2(xyz.shape[1])
    index_poses.add(poses)
    distances, ids = index_poses.search(poses_kf, 1)
    keyframe_indies = np.squeeze(ids)

    # if any index is smaller than n - 1, we need to set the index and position as the n_th frame
    invalid_indies = keyframe_indies < n - 1        # keyframe smaller than the sequence number
    keyframe_indies[invalid_indies] = n - 1
    poses_kf[invalid_indies, :] = poses[n - 1, :]

    return keyframe_indies, poses_kf


if __name__ == '__main__':
    config_path = '../configs/plot_temp.yaml'
    config = yaml.safe_load(open(config_path))

    seq = 'royce_hall'
    poses_folder_path = config['data_root']['poses']
    descriptors_folder_path = config['data_root']['descriptors']
    keyframes_folder_path = config['data_root']['keyframes']
    test_frames_folder_path = config['data_root']['test_frames']

    poses_path = os.path.join(poses_folder_path, seq, 'poses.txt')
    keyframe_poses_path = os.path.join(keyframes_folder_path, seq, 'poses/poses_kf.txt')

    descriptors_path = os.path.join(descriptors_folder_path, seq, 'test_whole_frame_descriptors.npy')
    keyframe_descriptors_path = os.path.join(descriptors_folder_path, seq, 'keyframe_descriptors.npy')
    test_frame_poses_path = os.path.join(test_frames_folder_path, seq, 'poses/poses.txt')

    n = 5   # seq size

    xyz, _ = load_xyz_rot(poses_path)
    xyz_kf, _ = load_xyz_rot(keyframe_poses_path)

    descriptors = load_descriptors(descriptors_path)
    descriptors_kf = load_descriptors(keyframe_descriptors_path)

    kf_indices, xyz_kf = find_keyframe_indices(xyz, xyz_kf, n)
    descriptors_seq, descriptors_kf_seq = stack_descriptors(descriptors, kf_indices, n)

    test_selection = 10
    # select 1 sample per test_selection samples, reduce the test size
    test_frame_descriptors = descriptors_seq[::test_selection]

    # regenerate the keyframe poses file (!!! has bug here, rotation not fix !!!)
    keyframe_poses_copy = np.loadtxt(keyframe_poses_path, delimiter=' ', dtype=np.float32)
    keyframe_poses_copy[0, [3, 7, 11]] = xyz_kf[0]

    # regenerate the test frame poses file (delete the first n-1 line)
    test_frame_poses_copy = np.loadtxt(test_frame_poses_path, delimiter=' ', dtype=np.float32)
    test_frame_poses_copy = test_frame_poses_copy[n-1:, :]

    np.save(os.path.join(descriptors_folder_path, seq, 'keyframe_seq_descriptors'), descriptors_kf_seq)
    np.save(os.path.join(descriptors_folder_path, seq, 'test_frame_seq_descriptors'), test_frame_descriptors)
    np.savetxt(os.path.join(keyframes_folder_path, seq, 'poses/poses_kf_seq.txt'), keyframe_poses_copy)
    np.savetxt(os.path.join(test_frames_folder_path, seq, 'poses/poses_seq.txt'), test_frame_poses_copy)
