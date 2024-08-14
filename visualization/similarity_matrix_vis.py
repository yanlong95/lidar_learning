import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import tqdm

from tools.fileloader import load_descriptors


def compute_distance_matrix(descriptors):
    num = descriptors.shape[0]
    distance_matrix = np.zeros((num, num))

    for i in tqdm.tqdm(range(num)):
        for j in range(num):
            distance_matrix[i, j] = (1.0 - np.dot(descriptors[i, :], descriptors[j, :])) / 2.0

    return distance_matrix


if __name__ == '__main__':
    descriptors_path = '/home/vectr/Desktop/temp_desc/mout-loop-1_descriptors.txt'
    descriptors = load_descriptors(descriptors_path)
    distance_matrix = compute_distance_matrix(descriptors)
    similarity_matrix = 1.0 - distance_matrix

    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    im = ax.imshow(similarity_matrix, cmap='viridis')
    fig.colorbar(im, cax=cax, orientation='vertical')
    plt.show()

