import tensorflow as tf

from configs.config import Config
from utils.dirs import create_dirs
from datasets.dataset import Dataset
from utils.augmentation import Augmentaton
from loaders.data_loader import DataLoader
from models.densenet import DenseNet
from utils.logger import Logger
from trainers.trainer import DenseNetTrainer


def main():
    config = Config()

    create_dirs([config.summary_dir, config.checkpoint_dir])

    sess = tf.Session()

    train_data = Dataset(config.root, config.train_image_file, config.type,
                         transform=Augmentaton(size=config.resize, mean=config.means[config.type],
                                               std=config.stds[config.type]), max_samples=None)
    valid_data = Dataset(config.root, config.valid_image_file, config.type,
                         transform=Augmentaton(size=config.resize, mean=config.means[config.type],
                                               std=config.stds[config.type]), max_samples=None)
    train_data_loader = DataLoader(train_data)
    valid_data_loader = DataLoader(valid_data)

    model = DenseNet(config)

    logger = Logger(sess, config)

    trainer = DenseNetTrainer(sess, model, train_data_loader, valid_data_loader, config, logger)

    model.load(sess)

    if config.phase == "train":
        trainer.train()

    elif config.phase == "test":
        trainer.test("prediction.csv")


if __name__ == '__main__':
    main()
