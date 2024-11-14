import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import ticker
from tools.fileloader import load_descriptors


def add_2d_scatter(ax, points, points_color, title=None, alpha=0.5):
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=50, alpha=alpha)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())


if __name__ == '__main__':
    descriptors_path = '/home/vectr/Desktop/temp_desc/parkland_mount.txt'
    descriptors = load_descriptors(descriptors_path)

    t_sne = TSNE(
        n_components=2,
        perplexity=30,
        init="random",
        random_state=0,
    )

    descriptors_tsne = t_sne.fit_transform(descriptors)
    descriptors_2d = descriptors_tsne[:len(descriptors), :]
    colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(descriptors)))

    # plot
    fig, ax1 = plt.subplots(1, 1, figsize=(3, 3), facecolor="white", constrained_layout=True)
    add_2d_scatter(ax1, descriptors_tsne, colors, alpha=0.5)
    plt.show()
