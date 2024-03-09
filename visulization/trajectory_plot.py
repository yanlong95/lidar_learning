import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def trajectory_plot(poses, keyframe_poses):
    # plt.figure()
    # norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)
    # mapper = matplotlib.cm.ScalarMappable(norm=norm)
    # mapper.set_array(test_frame_overlap)
    # colors = np.array([mapper.to_rgba(a) for a in test_frame_overlap])
    #
    # plt.scatter(test_frame_poses_sorted[:, 0], test_frame_poses_sorted[:, 1], c=colors[test_frame_indices], s=10)
    # plt.scatter(keyframe_poses[:, 0], keyframe_poses[:, 1], c='tan', s=5, label='keyframes')
    # plt.scatter(top_n_keyframe_poses[:, 0], top_n_keyframe_poses[:, 1], c='magenta', s=5, label='top n choices')
    # plt.scatter(test_frame_poses[idx, 0], test_frame_poses[idx, 1], c='orange', s=20, label='current location')

    # plt.axis('square')
    # plt.xlabel('X [m]')
    # plt.ylabel('Y [m]')
    # plt.title('Overlap Map')
    # plt.legend()
    # cbar = plt.colorbar(mapper)
    # cbar.set_label('Overlap', rotation=270, weight='bold')
    #
    # plt.show()
    pass


if __name__ == '__main__':
    descriptors_path = ''
    keyframes_poses_path = ''
    poses_path = ''
