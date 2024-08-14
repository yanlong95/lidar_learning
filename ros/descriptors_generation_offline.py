import torch
import numpy as np
import tqdm

from tools.fileloader import load_files, read_image
from modules.overlap_transformer import OverlapTransformer32


def compute_descriptors(img_paths, model, descriptor_size=256):
    num_scans = len(img_paths)
    descriptors = np.zeros((num_scans, descriptor_size), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for i in tqdm.tqdm(range(num_scans)):
            curr_batch = read_image(img_paths[i])
            curr_batch = torch.cat((curr_batch, curr_batch), dim=0)
            curr_descriptors = model(curr_batch)
            descriptors[i, :] = curr_descriptors[0, :].cpu().detach().numpy()

    descriptors = descriptors.astype(np.float32)
    return descriptors

if __name__ == '__main__':
    # load model
    weights_path = '/media/vectr/vectr3/Dataset/overlap_transformer/weights/weights_07_01/best.pth.tar'
    checkpoint = torch.load(weights_path)
    model = OverlapTransformer32().to('cuda')
    model.load_state_dict(checkpoint['state_dict'])

    # load image paths
    img_folder = '/media/vectr/vectr7/arl/png_files/mout-loop-1/512'
    img_paths = load_files(img_folder)
    descriptors = compute_descriptors(img_paths, model)

    # save descriptors
    save_path = '/home/vectr/Desktop/temp_desc/mout-loop-1_descriptors.txt'
    np.savetxt(save_path, descriptors)
