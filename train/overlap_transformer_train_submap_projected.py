"""
Code taken from https://github.com/haomo-ai/OverlapTransformer/. The framework has been rewritten to avoid the for
loop searching which used to find the anchor-positive-negative batch.
"""
# p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
# if p not in sys.path:
#     sys.path.append(p)
# sys.path.append('../tools/')
# sys.path.append('../modules/')

import os
import sys
import tqdm
import yaml
import torch
import numpy as np
from tensorboardX import SummaryWriter
from modules.overlap_transformer_submap import OverlapTransformer32Submap
from modules.losses.overlap_transformer_loss import triplet_loss, triplet_confidence_loss
from valid.overlap_transformer_valid_submap_projected import validation
from tools.read_datasets_submap_projected import (overlaps_submaps_loader, read_one_batch_overlaps_submap_n2n,
                                        read_one_batch_overlaps_submap_one2n)
from tools.utils import RunningAverage, save_checkpoint, load_checkpoint


class trainHandler():
    def __init__(self, params=None, img_folder=None, overlaps_folder=None, submaps_folder=None, weights_folder=None,
                 resume=False):
        super(trainHandler, self).__init__()

        # define model, optimizer, and scheduler
        self.height = params['height']
        self.width = params['width']
        self.channels = params['channels']
        self.use_transformer = params['use_transformer']

        self.num_epochs = params['num_epochs']
        self.num_pos_max = params['num_pos_max']
        self.num_neg_max = params['num_neg_max']
        self.learning_rate = params['learning_rate']
        self.margin1 = params['margin1']
        self.alpha = params['alpha']
        self.metric = params['metric']
        self.submap_size = params['submap_size']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = OverlapTransformer32Submap(width=self.width, channels=self.channels, submap_size=self.submap_size,
                                                use_transformer=self.use_transformer).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.3)

        # load directories
        self.img_folder = img_folder                    # images path
        self.overlaps_folder = overlaps_folder          # overlaps path (label path)
        self.submaps_folder = submaps_folder            # submaps path
        self.weights_folder = weights_folder            # weight path

        # resume training
        self.resume = resume
        self.restore_path = os.path.join(self.weights_folder, 'last.pth.tar')

    def train(self):
        # set train mode
        self.model.train()

        # count losses for each epoch
        loss_avg = RunningAverage()

        # load all training pairs
        overlaps_data, anchor_submaps, pos_neg_submaps = overlaps_submaps_loader(self.overlaps_folder,
                                                                                 self.submaps_folder, shuffle=True)
        num_scans = overlaps_data.shape[0]

        for j in tqdm.tqdm(range(num_scans)):
            # load a batch for a single scan with overlaps values
            anchor_batch, pos_batch, neg_batch, pos_overlaps, neg_overlaps, num_pos, num_neg = (
                read_one_batch_overlaps_submap_n2n(self.img_folder, self.submaps_folder, overlaps_data, anchor_submaps,
                                                   pos_neg_submaps, j, self.channels, self.height, self.width,
                                                   self.num_pos_max, self.num_neg_max, self.submap_size, self.device,
                                                   shuffle=True))

            # in case no pair
            if num_pos == 0 or num_neg == 0:
                continue

            input_batch = torch.cat((anchor_batch, pos_batch, neg_batch), dim=0)

            output_batch = self.model(input_batch)
            o1, o2, o3 = torch.split(output_batch, [1, num_pos, num_neg], dim=0)
            # loss = triplet_loss(o1, o2, o3, self.margin1, lazy=False, ignore_zero_loss=True)
            loss = triplet_confidence_loss(o1, o2, o3, pos_overlaps, self.margin1, alpha=self.alpha,
                                           lazy=False, ignore_zero_loss=True, metric=self.metric)

            if loss == -1:
                continue

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_avg.update(loss.item())

        avg_loss = loss_avg()
        return avg_loss

    def train_eval(self):
        epochs = self.num_epochs

        if self.resume:
            epoch, best_val = load_checkpoint(self.restore_path, self.model, self.optimizer)
            start_epoch = epoch + 1
            train_start_str = f"Resuming from {self.restore_path}."

        else:
            start_epoch = 1
            best_val = 0.0
            train_start_str = "Training From Scratch."

        writer1 = SummaryWriter(comment=f"LR_{self.learning_rate}_{self.metric}_alpha_{self.alpha}_schedule_margin1_{self.margin1}")

        overlaps_data, _, _ = overlaps_submaps_loader(self.overlaps_folder, self.submaps_folder, shuffle=True)
        print("=======================================================================\n")
        print(train_start_str)
        print("Training with seq: ", np.unique(overlaps_data[:, 1]))
        print("Total pairs: ", np.sum(overlaps_data[:, 3]) + np.sum(overlaps_data[:, 4]))
        print("\n=======================================================================")

        for epoch in range(start_epoch, start_epoch+epochs):
            print(f'training with epoch: {epoch}')
            loss = self.train()
            self.scheduler.step()
            writer1.add_scalar("train_losses", loss, global_step=epoch)

            # validate model
            with torch.no_grad():
                loss_valid, topn_rate, topn_rate_dist = validation(self.model, top_n=5, metric=self.metric, method='overlap')
                loss_valid_2, topn_rate_2, topn_rate_dist_2 = validation(self.model, top_n=5, metric=self.metric, method='overlap', seq=1)
                loss_valid_3, topn_rate_3, topn_rate_dist_3 = validation(self.model, top_n=5, metric=self.metric, method='overlap', seq=2)

                # writer1.add_scalar("topn_rate", topn_rate, global_step=epoch)
                # writer1.add_scalar("valid_losses", loss_valid, global_step=epoch)
                # writer1.add_scalar("topn_rate_dist", topn_rate_dist, global_step=epoch)
                #
                # writer1.add_scalar("topn_rate_2", topn_rate_2, global_step=epoch)
                # writer1.add_scalar("valid_losses_2", loss_valid_2, global_step=epoch)
                # writer1.add_scalar("topn_rate_dist_2", topn_rate_dist_2, global_step=epoch)
                #
                # writer1.add_scalar("topn_rate_3", topn_rate_3, global_step=epoch)
                # writer1.add_scalar("valid_losses_3", loss_valid_3, global_step=epoch)
                # writer1.add_scalar("topn_rate_dist_3", topn_rate_dist_3, global_step=epoch)

                writer1.add_scalars("topn_rate", {'geo': topn_rate, 'royce': topn_rate_2,
                                                  'sculpture': topn_rate_3}, global_step=epoch)
                writer1.add_scalars("topn_rate_dist", {'geo': topn_rate_dist,
                                                       'royce': topn_rate_dist_2, 'sculpture': topn_rate_dist_3},
                                    global_step=epoch)
                writer1.add_scalars("valid_loss", {'geo': loss_valid, 'royce': loss_valid_2,
                                                   'sculpture': loss_valid_3}, global_step=epoch)

            # check if current model has the best validation rate
            if topn_rate >= best_val:
                best_val = topn_rate
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
    # load config
    # ================================================================
    config_filename = '/home/vectr/PycharmProjects/lidar_learning/configs/config.yml'
    parameters_path = '/home/vectr/PycharmProjects/lidar_learning/configs/parameters.yml'

    config = yaml.safe_load(open(config_filename))
    params = yaml.safe_load(open(parameters_path))

    data_root = config["data_root"]["root"]
    train_img_folder = config["data_root"]["png_files"]
    train_overlaps_folder = config["data_root"]["gt_overlaps"]
    train_submaps_folder = config["data_root"]["submaps"]
    weights_folder = config["data_root"]["weights"]
    train_seqs = config["seqs"]["train"]
    parameters = params['learning']
    # ============================================================================

    train_overlaps_paths = [os.path.join(train_overlaps_folder, seq, 'overlaps_train.npz') for seq in train_seqs]
    train_submaps_paths = [os.path.join(train_submaps_folder, seq) for seq in train_seqs]
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
    train_handler = trainHandler(params=parameters, img_folder=train_img_folder, overlaps_folder=train_overlaps_paths,
                                 submaps_folder=train_submaps_paths, weights_folder=weights_folder, resume=False)
    train_handler.train_eval()
