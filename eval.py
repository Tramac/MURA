from __future__ import division

import argparse
import sys

import tensorflow as tf

from datasets.dataset import Dataset
from utils.augmentation import Augmentaton
from loaders.data_loader import DataLoader
from models.densenet import DenseNet
from configs.config import Config
from utils.logger import Logger
from trainers.trainer import DenseNetTrainer


# parser = argparse.ArgumentParser()
# parser.add_argument('--valid_image_paths', type=str, default="MURA-v1.1/valid_image_paths.csv", help="Valid datasets")
# parser.add_argument('--output_prediction_path', type=str, default="output_prediction_paths.csv", help="prediction result path")
#
# args = parser.parse_args()
valid_image_paths = sys.argv[1]
output_prediction_path = sys.argv[2]


def evaluate():
    config = Config()
    valid_data = Dataset(config.root, valid_image_paths, config.type,
                         transform=Augmentaton(size=config.resize, mean=config.means[config.type],
                                               std=config.stds[config.type]), max_samples=10)
    valid_data_loader = DataLoader(valid_data)

    sess = tf.Session()
    model = DenseNet(config)
    logger = Logger(sess, config)
    trainer = DenseNetTrainer(sess, model, valid_data_loader, valid_data_loader, config, logger)

    model.load(sess)

    if config.phase == "train":
        trainer.train()

    elif config.phase == "test":
        trainer.test(output_prediction_path)

if __name__ == '__main__':
    evaluate()
