from __future__ import division

import os


class Config(object):
    exp_name = "MURA"
    summary_dir = os.path.join("./experiments", exp_name, "summary/")
    checkpoint_dir = os.path.join("./experiments", exp_name, "checkpoint/")
    root = "./"
    # root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    train_image_file = "MURA-v1.1/train_image_paths.csv"
    valid_image_file = "MURA-v1.1/valid_image_paths.csv"
    type = "SHOULDER"
    phase = "train"
    resize = 128
    means = {"SHOULDER": 65.976}
    stds = {"SHOULDER": 49.540}
    input_shape = [128, 128, 1]
    n_classes = 2
    n_blocks = 4
    layers_per_block = 6
    growth_rate = 12
    dropout_rate = 0.2
    reduction = 0.5
    weight_decay = 1e-4
    max_to_keep = 3
    num_epochs = 50
    reduce_lr_epoch_1 = 25 # 0.5 * num_epoch
    reduce_lr_epoch_2 = 40 # 0.75 * num_epochs
    learning_rate = 0.01
    batch_size = 16
    valid_num_epoch = 1
