import os
import yaml
import numpy as np
from sklearn.manifold import TSNE
import sklearn.datasets as datasets
import matplotlib.pyplot as plt
from matplotlib import ticker
from tools.fileloader import load_descriptors, load_xyz_rot


def add_2d_scatter(ax, points, points_color, title=None, alpha=0.5):
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=50, alpha=alpha)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())


if __name__ == '__main__':
    seq = 'sculpture_garden'
    config = yaml.safe_load(open('/home/vectr/PycharmProjects/lidar_learning/configs/config.yml'))
    descriptors_path = os.path.join(config['data_root']['descriptors'], f'{seq}/descriptors.npy')
    descriptors_kf_path = os.path.join(config['data_root']['descriptors'], f'{seq}/descriptors_kf.npy')
    poses_path = os.path.join(config['data_root']['poses'], f'{seq}/poses.txt')
    poses_kf_path = os.path.join(config['data_root']['keyframes'], f'{seq}/poses/poses_kf.txt')
    descriptors = load_descriptors(descriptors_path)
    descriptors_kf = load_descriptors(descriptors_kf_path)
    xyz, _ = load_xyz_rot(poses_path)
    xyz_kf, _ = load_xyz_rot(poses_kf_path)

    t_sne = TSNE(
        n_components=2,
        perplexity=30,
        init="random",
        random_state=0,
    )

    descriptors_tsne = t_sne.fit_transform(np.concatenate((descriptors, descriptors_kf), axis=0))
    descriptors_2d = descriptors_tsne[:len(descriptors), :]
    descriptors_kf_2d = descriptors_tsne[len(descriptors):, :]
    colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(descriptors)))
    colors_kf = plt.get_cmap('plasma')(np.linspace(0, 1, len(descriptors_kf)))
    # plot_2d(descriptors_2d, descriptors_kf_2d, colors)

    # plot
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(3, 3), facecolor="white", constrained_layout=True)
    fig.suptitle(f'{seq}', size=16)
    add_2d_scatter(ax1, descriptors_2d, colors, alpha=0.5)
    add_2d_scatter(ax1, descriptors_kf_2d, colors_kf, alpha=1.0)

    ax2.scatter(xyz[:, 0], xyz[:, 1], c=colors, alpha=0.5)
    ax2.scatter(xyz_kf[:, 0], xyz_kf[:, 1], c=colors_kf, alpha=1.0)
    plt.show()
