import os
import tqdm
import yaml
import cv2

from projection import RangeProjection
from range_view_loader import RangeViewLoader
from parser import SemanticKitt
from tools.fileloader import load_files

config_path = '/home/vectr/PycharmProjects/lidar_learning/data/kitti/dataset/config_kitti.yml'
config = yaml.safe_load(open(config_path))

data_root = config['data_root']
sequences = config['sequences']

num_aug = 10

for sequence in sequences:
    kitti = SemanticKitt(data_root, [sequence])
    dataloader = RangeViewLoader(dataset=kitti, config=config, is_train=True, depth_only=True, normalize=False)

    for j in tqdm.tqdm(range(num_aug)):
        dst_path = os.path.join(data_root, sequence, 'png_files/512', f'aug{j}')
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        for i in range(len(dataloader)):
            img = dataloader[i].to('cpu').numpy()
            img = img[0, :, :]
            filename = os.path.join(dst_path, f'{str(i).zfill(6)}.png')
            cv2.imwrite(filename, img)
