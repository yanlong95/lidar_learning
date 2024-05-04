import os
import logging
import json
import shutil

import torch
import numpy as np


class Params:
    """
    Loads hyperparameters from a json file.
    """
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        # save parameters to json file
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        # update parameters from json file
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        # Access Params instance by `params.dict['learning_rate']
        return self.__dict__


class RunningAverage:
    """
    A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def save_dict_to_json(d, json_path):
    """
    save dict to json file

    Args:
        d: (dict) dictionary to save
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint):
    """
    Saves model and training parameters at checkpoint + 'last.pth.tar'.
    If is_best is True, save checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contain epoch, model_static_dict, optimizer_static_dict and loss
        is_best: (bool) True if the current model has the best performance
        checkpoint: (string) folder where the parameters are saved
    """
    file_path = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print(f'Checkpoint not found! Creating new folder {checkpoint}.')
        os.makedirs(checkpoint)
    else:
        print('Checkpoint found!')

    torch.save(state, file_path)

    if is_best:
        shutil.copyfile(file_path, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """
    Loads model and training parameters from checkpoint folder.

    Args:
        checkpoint: (string) folder where the model parameters are
        model: (torch.nn.Module) the model to be loaded
        optimizer: (torch.optim) resume the optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise FileNotFoundError('Checkpoint not found!')

    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint


def set_logger(log_path):
    """
    Set the logger to log info in terminal and file `log_path`.

    Args:
        log_path: (string) where to log
    Returns:
        the logger
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # logging to a file
        file_formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # logging to console
        stream_formatter = logging.Formatter('%(message)s')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

    return logger
