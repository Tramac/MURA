from __future__ import division

import numpy as np


class DataLoader(object):
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, dataset):
        self.dataset = dataset

    def next_batch(self, batch_size):
        self.images = []
        self.labels = []
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > len(self.dataset):
            self.epochs_completed += 1
            self.dataset.shuffle_images_and_labels()
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        for idx in range(start, end):
            image, label, _ = self.dataset[idx]
            self.images.append(image)
            self.labels.append(label)

        return self.ToArray(self.images, self.labels)

    def get_random_single_sample(self):
        indexes = np.random.randint(0, len(self.dataset))
        image, label, _ = self.dataset[indexes]
        return image, label

    def get_single_sample(self, index):
        image, label, name = self.dataset[index]
        image = np.array(image, dtype=np.float32)
        label = np.array([label], dtype=np.int32)
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=3)

        return image, label, name


    def ToArray(self, images, labels):
        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)

        images = np.expand_dims(images, axis=3)

        return images, labels
