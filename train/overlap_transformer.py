"""
train the model overlapTransformer.
"""

import os
import sys
import yaml
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from modules.ot_copy.tools.read_all_sets import overlap_orientation_npz_file2string_string_nparray
from modules.ot_copy.modules.overlap_transformer_haomo import featureExtracter
from modules.ot_copy.tools.read_samples import read_one_batch_pos_neg
from modules.ot_copy.tools.read_samples import read_one_need_from_seq
import modules.ot_copy.modules.loss as PNV_loss
from modules.ot_copy.tools.utils.utils import *
from modules.ot_copy.valid.valid_seq_os1_rewrite_new import validation


class TrainHandler:
    def __init__(self, height=32, width=900, channel=1, lr=0.001, norm_layer=None, use_transformer=True):
        self.height = height
        self.width = width
        self.channel = channel
        self.lr = lr
        self.norm_layer = norm_layer
        self.use_transformer = use_transformer

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.amodel = featureExtracter(height, width, channel, norm_layer, use_transformer).to(self.device)
        self.parameters = self.amodel.parameters()
        self.optimizer = torch.optim.Adam(self.parameters, self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=1.0)