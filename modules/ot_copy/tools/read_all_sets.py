import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
    
import numpy as np

"""Use the tools from OverlapNet"""

def overlap_orientation_npz_file2string_string_nparray(npzfilenames, shuffle=True):

    imgf1_all = []
    imgf2_all = []
    dir1_all = []
    dir2_all = []
    overlap_all = []
    overlap_all_threshold = []

    for npzfilename in npzfilenames:
        h = np.load(npzfilename, allow_pickle=True)
        imgf1 = np.char.mod('%06d', h['overlaps'][:, 0]).tolist()
        imgf2 = np.char.mod('%06d', h['overlaps'][:, 1]).tolist()
        overlap = h['overlaps'][:, 2]
        overlap_thresh = h['overlaps'][:, 3]
        dir1 = (h['seq'][:, 0]).tolist()
        dir2 = (h['seq'][:, 1]).tolist()

        if shuffle:
            shuffled_idx = np.random.permutation(overlap.shape[0])
            imgf1 = (np.array(imgf1)[shuffled_idx]).tolist()
            imgf2 = (np.array(imgf2)[shuffled_idx]).tolist()
            dir1 = (np.array(dir1)[shuffled_idx]).tolist()
            dir2 = (np.array(dir2)[shuffled_idx]).tolist()
            overlap = overlap[shuffled_idx]
            overlap_thresh = overlap_thresh[shuffled_idx]

        imgf1_all.extend(imgf1)
        imgf2_all.extend(imgf2)
        dir1_all.extend(dir1)
        dir2_all.extend(dir2)
        overlap_all.extend(overlap)
        overlap_all_threshold.extend(overlap_thresh)

    return (imgf1_all, imgf2_all, dir1_all, dir2_all, np.asarray(overlap_all), np.asarray(overlap_all_threshold))


if __name__ == '__main__':
    train_dataset = ['/home/vectr/Documents/Dataset/train/botanical_garden/overlaps/train_set_reduced.npz']
    train_imgf1, train_imgf2, train_dir1, train_dir2, train_overlap, train_overlap_threshold = \
        overlap_orientation_npz_file2string_string_nparray(train_dataset, shuffle=True)
