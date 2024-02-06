import yaml
import os
import logging
import argparse
import torch

from tools import utils


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='./model', help='Directory contain params.json')
parser.add_argument('--reestore_file', default=None, help='Optional, name of the file in --model_dir '
                                                          'containing weights to reload before training')

if __name__ == "__main__":
    config_path = './config/config.yml'
    config = yaml.safe_load(open(config_path))
    data_root = config['data_root']['data_root_folder']

    # load learning hyperparameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'parameters.json')
    params = utils.Params(json_path)

    # check GPU
    params.cuda = torch.cuda.is_available()

    # set random seed for repeating experiment
    torch.manual_seed(0)
    if params.cuda:
        torch.cuda.manual_seed(0)

    # set logger
    logger = utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # create the training pipeline
    logger.info("Starting training")
