from __future__ import division

import os
import random

from re import findall
from scipy.misc import imread


class Dataset(object):
    def __init__(self, root, images_file, type, transform=None, max_samples=None):
        self.root = root
        self.type = type
        self.means, self.stds = self.measure_mean_and_std()
        self.transofm = transform
        self.images_and_labels = list()
        with open(os.path.join(self.root, images_file), "r") as f:
            for line in f.readlines():
                # if findall(type, line):
                    if findall("positive", line):
                        self.images_and_labels.append({"image": line.strip(), "label": int(1)})
                    elif findall("negative", line):
                        self.images_and_labels.append({"image": line.strip(), "label": int(0)})

        if "train" in images_file:
            print("Train dataset, number of samples: ", len(self.images_and_labels))
            self.shuffle_images_and_labels()

        if "valid" in images_file:
            print("Valid dataset, number of samples: ", len(self.images_and_labels))

        if max_samples:
            self.images_and_labels = self.images_and_labels[:max_samples]

    def measure_mean_and_std(self):
        # SHOULDER mean: 65.976 std: 49.540
        '''
        sum_mean = 0
        sum_std = 0
        pixels_num = 0

        for idx, img in enumerate(self.images_and_labels):
            print("Loading...{}/{}".format(idx, len(self.images_and_labels)))
            image = imread(os.path.join(self.root, img["image"]))
            if len(image.shape) == 3:
                image = image[:, :, 0]
            height, width = image.shape[0], image.shape[1]
            sum_mean += np.sum(image)
            pixels_num += (height * width)
        mean = sum_mean / pixels_num

        for idx, img in enumerate(self.images_and_labels):
            print("Loading...{}/{}".format(idx, len(self.images_and_labels)))
            image = imread(os.path.join(self.root, img["image"]))
            if len(image.shape) == 3:
                image = image[:, :, 0]
            sum_std += np.sum(np.square(image - mean))
        std = np.sqrt(sum_std / pixels_num)
        '''
        means = {"SHOULDER": 65.976}
        stds = {"SHOULDER": 49.540}

        return means, stds

    @property
    def images_mean(self):
        return self.means[self.type]

    @property
    def images_std(self):
        return self.stds[self.type]

    def __getitem__(self, index):
        image, label, name = self.pull_item(index)

        return image, label, name

    def __len__(self):
        return len(self.images_and_labels)

    def pull_item(self, index):
        image_and_label = self.images_and_labels[index]

        name = image_and_label["image"]
        image = imread(os.path.join(self.root, image_and_label["image"]))
        if len(image.shape) == 3:
            image = image[:, :, 0]
        label = image_and_label["label"]

        if self.transofm is not None:
            image, label = self.transofm(image, label)

        return image, label, name

    def shuffle_images_and_labels(self):
        random.shuffle(self.images_and_labels)
