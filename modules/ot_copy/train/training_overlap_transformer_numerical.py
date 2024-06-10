#!/usr/bin/env python3
# Developed by Junyi Ma, Xieyuanli Chen, and Jun Zhang
# This file is covered by the LICENSE file in the root of the project OverlapTransformer:
# https://github.com/haomo-ai/OverlapTransformer/
# Brief: train OverlapTransformer with KITTI sequences

import os
import sys
import tqdm

p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
sys.path.append('../tools/')
sys.path.append('../modules/')    
import torch
import numpy as np
from tensorboardX import SummaryWriter
from modules.ot_copy.modules.overlap_transformer_haomo import featureExtracter
np.set_printoptions(threshold=sys.maxsize)
import modules.ot_copy.modules.loss as PNV_loss
from modules.ot_copy.tools.utils.utils import *
from modules.ot_copy.valid.valid_seq_os1_rewrite_new import validation
import yaml

from modules.ot_copy.tools.read_all_sets_reformat import overlaps_loader, read_one_batch_pos_neg_numerical
from tools.utils import RunningAverage, save_checkpoint


class trainHandler():
    def __init__(self, params=None, img_folder=None, overlaps_folder=None, weights_folder=None):
        super(trainHandler, self).__init__()

        # define model, optimizer, and scheduler
        self.height = params['height']
        self.width = params['width']
        self.channels = params['channels']
        self.use_transformer = params['use_transformer']

        self.num_epochs = params['num_epochs']
        self.num_pos_max = params['num_pos_max_overlap']
        self.num_neg_max = params['num_neg_max_overlap']
        self.margin1 = params['margin1']
        self.learning_rate = params['learning_rate_overlap']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = featureExtracter(width=self.width, channels=self.channels, use_transformer=self.use_transformer).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.5)

        # load directories
        self.img_folder = img_folder                    # images path
        self.overlaps_folder = overlaps_folder          # overlaps path (label path)
        self.weights_folder = weights_folder            # weight path

        # resume training
        self.resume = True
        self.restore_path = os.path.join(self.weights_folder, 'last.pth.tar')

    def train(self):
        # set train mode
        self.model.train()

        # count loss for each epoch
        loss_avg = RunningAverage()

        # load all training pairs
        overlaps_data = overlaps_loader(self.overlaps_folder, shuffle=True)
        num_scans = overlaps_data.shape[0]

        for j in tqdm.tqdm(range(num_scans)):
            # load a batch for a single scan
            (anchor_batch, pos_batch, neg_batch, pos_batch_overlaps, neg_batch_overlaps, num_pos, num_neg) = \
                (read_one_batch_pos_neg_numerical(self.img_folder, overlaps_data, j, self.channels, self.height,
                                                  self.width, self.num_pos_max, self.num_neg_max, self.device,
                                                  shuffle=True))

            # in case no pair
            if num_pos == 0 or num_neg == 0:
                continue

            input_batch = torch.cat((anchor_batch, pos_batch, neg_batch), dim=0)
            output_batch = self.model(input_batch)
            o1, o2, o3 = torch.split(output_batch, [1, num_pos, num_neg], dim=0)
            loss = PNV_loss.mean_squared_error_loss(o1, o2, o3, pos_batch_overlaps, neg_batch_overlaps, alpha=1.0)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_avg.update(loss.item())

        avg_loss = loss_avg()
        return avg_loss

    def train_eval(self):
        epochs = self.num_epochs

        if self.resume:
            checkpoint = torch.load(self.restore_path)
            start_epoch = checkpoint['epoch'] + 1
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            best_val = checkpoint['best_val']
            print(f"Resuming from {self.restore_path}.")

        else:
            print("Training From Scratch ...")
            start_epoch = 0
            best_val = sys.maxsize

        writer1 = SummaryWriter(comment=f"LR_{self.learning_rate}_overlap")

        overlaps_data = overlaps_loader(self.overlaps_folder, shuffle=True)
        print("=======================================================================\n\n")
        print("training with seq: ", np.unique(overlaps_data[:, 1]))
        print("total pairs: ", np.sum(overlaps_data[:, 3]) + np.sum(overlaps_data[:, 4]))
        print("\n\n=======================================================================")

        for epoch in range(start_epoch, start_epoch+epochs):
            print("\n=======================================================================")
            print(f'training with epoch: {epoch}')
            loss = self.train()
            self.scheduler.step()
            writer1.add_scalar("loss_train", loss, global_step=epoch)

            # validate model
            with torch.no_grad():
                loss_val = validation(self.model, top_n=5)
                writer1.add_scalar("loss_val", loss_val, global_step=epoch)

            # check if current model has the best validation rate
            if loss_val <= best_val:
                best_val = loss_val
                is_best = True
            else:
                is_best = False

            # save model
            print("saving weights ...")
            save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_val': best_val},
                is_best=is_best,
                checkpoint=self.weights_folder
            )


if __name__ == '__main__':
    # load config ================================================================
    config_filename = '/home/vectr/PycharmProjects/lidar_learning/configs/config.yml'
    parameters_path = '/home/vectr/PycharmProjects/lidar_learning/configs/parameters.yml'

    config = yaml.safe_load(open(config_filename))
    params = yaml.safe_load(open(parameters_path))

    range_images_folder = config["data_root"]["png_files"]
    gt_overlaps_folder = config["data_root"]["gt_overlaps"]
    weights_folder = config["data_root"]["weights"]
    train_seqs = config["seqs"]["train"]
    parameters = params['learning']
    # ============================================================================

    train_overlaps_folder = [os.path.join(gt_overlaps_folder, seq, 'overlaps_train.npz') for seq in train_seqs]
    train_img_folder = os.path.join(range_images_folder, '512')
    # model_dir = '/home/vectr/PycharmProjects/lidar_learning/model'

    """
        trainHandler to handle with training process.
        Args:
            height: the height of the range image (the beam number for convenience).
            width: the width of the range image (900, alone the lines of OverlapNet).
            channels: 1 for depth only in our work.
            norm_layer: None in our work for better model.
            use_transformer: Whether to use MHSA.
            lr: learning rate, which needs to fine tune while training for the best performance.
            data_root_folder: root of KITTI sequences. It's better to follow our file structure.
            train_set: traindata_npzfiles (alone the lines of OverlapNet).
            training_seqs: sequences number for training (alone the lines of OverlapNet).
    """
    train_handler = trainHandler(params=parameters, img_folder=train_img_folder, overlaps_folder=train_overlaps_folder,
                                 weights_folder=weights_folder)
    train_handler.train_eval()
