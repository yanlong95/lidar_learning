import os
import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.io import read_image


def overlaps_loader(overlaps_paths, shuffle=True):
    overlaps_all = []
    for overlaps_path in overlaps_paths:
        overlaps = np.load(overlaps_path, allow_pickle=True)['overlaps']
        overlaps_all.extend(overlaps)

    overlaps_all = np.asarray(overlaps_all)
    if shuffle:
        np.random.shuffle(overlaps_all)

    return overlaps_all


class OverlapsDataset:
    def __init__(self, img_dir, overlaps_dir, train=True, transform=None):
        self.data_dir = img_dir
        self.label_dir = overlaps_dir
        self.train = train
        self.transform = transform

        if self.train:
            self.overlaps = overlaps_loader(self.label_dir, shuffle=True)

    def __len__(self):
        return len(self.overlaps)

    def __getitem__(self, idx):
        img1_id, img2_id, seq, overlap = self.overlaps[idx]
        img1_path = os.path.join(self.data_dir, seq, f'{str(img1_id).zfill(6)}.png')
        img2_path = os.path.join(self.data_dir, seq, f'{str(img2_id).zfill(6)}.png')
        img1 = read_image(img1_path)
        img2 = read_image(img2_path)
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, overlap
