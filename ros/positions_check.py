import os
import numpy as np
import matplotlib.pyplot as plt
from tools.fileloader import load_files, load_xyz_rot


if __name__ == '__main__':
    frames_poses_path = '/media/vectr/vectr3/Dataset/arl/poses/out-and-back-3/poses.txt'
    keyframes_poses_path = '/media/vectr/vectr3/Dataset/arl/poses/out-and-back-3/poses_kf.txt'

    xyz, rot = load_xyz_rot(frames_poses_path)
    xyz_kf, rot_kf = load_xyz_rot(keyframes_poses_path)

    print(f'number of frames: {len(xyz)}')
    print(f'number of keyframes: {len(xyz_kf)}')

    skip = 7100
    skip_kf = 183
    # plt.scatter(xyz[:skip, 0], xyz[:skip, 1])
    # plt.scatter(xyz_kf[:skip_kf, 0], xyz_kf[:skip_kf, 1])
    # plt.show()

    xyz1 = xyz[:skip, :]
    rot1 = rot[:skip, :]
    xyz2 = xyz[skip:, :]
    rot2 = rot[skip:, :]
    xyz_kf1 = xyz_kf[:skip_kf, :]
    rot_kf1 = rot_kf[:skip_kf, :]
    xyz_kf2 = xyz_kf[skip_kf:, :]
    rot_kf2 = rot_kf[skip_kf:, :]

    t = xyz[skip, :]
    R = rot[skip, :]
    R_inv = np.linalg.inv(R)
    xyz2_inv = (xyz2 - t) @ R_inv

    # plt.scatter(xyz1[:, 0], xyz1[:, 1])
    # plt.scatter(xyz2[:, 0], xyz2[:, 1])
    # plt.scatter(xyz2_inv[:, 0], xyz2_inv[:, 1])
    # plt.show()

    # src_folder = '/media/vectr/vectr3/Dataset/arl/test_alignment/out-and-back-3/png_files/512_2'
    # src_files = load_files(src_folder)
    # num_src_files = len(src_files)
    # if num_src_files < skip:
    #     raise 'ERROR! Source folder do not have enough files'
    # elif num_src_files == skip:
    #     print('No file need to be deleted.')
    # else:
    #     print(f'Deleting {num_src_files - skip} files ...')
    #     # removed_files_paths = src_files[skip:]
    #     removed_files_paths = src_files[:skip]
    #     for removed_file_path in removed_files_paths:
    #         os.remove(removed_file_path)
    #     print('Done!')

    dst_folder = '/media/vectr/vectr3/Dataset/arl/test_alignment/out-and-back-3/poses'
    frames_file1_path = os.path.join(dst_folder, 'poses_1.txt')
    frames_file2_path = os.path.join(dst_folder, 'poses_2.txt')
    keyframes_file1_path = os.path.join(dst_folder, 'poses_1_kf.txt')
    frames_file2_gt_path = os.path.join(dst_folder, 'poses_2_gt.txt')

    poses1 = np.zeros((len(xyz1), 3, 4))
    poses1[:, :, :3] = rot1
    poses1[:, :, 3] = xyz1

    poses2 = np.zeros((len(xyz2), 3, 4))
    poses2[:, :, 3] = xyz2_inv
    poses2_gt = np.zeros((len(xyz2), 3, 4))
    poses2_gt[:, :, 3] = xyz2

    poses1_kf = np.zeros((len(xyz_kf1), 3, 4))
    poses1_kf[:, :, 3] = xyz_kf1

    np.savetxt(frames_file1_path, poses1.reshape(-1, 12))
    np.savetxt(frames_file2_path, poses2.reshape(-1, 12))
    np.savetxt(keyframes_file1_path, poses1_kf.reshape(-1, 12))
    np.savetxt(frames_file2_gt_path, poses2_gt.reshape(-1, 12))

    xyz1, _ = load_xyz_rot(frames_file1_path)
    xyz2, _ = load_xyz_rot(frames_file2_path)
    plt.scatter(xyz1[:, 0], xyz1[:, 1], c='blue')
    plt.scatter(xyz2[:, 0], xyz2[:, 1], c='red')
    plt.scatter(xyz_kf1[:, 0], xyz_kf1[:, 1], c='pink')
    plt.show()
