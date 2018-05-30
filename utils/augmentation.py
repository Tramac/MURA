from __future__ import division

import random
import numpy as np

from scipy.misc import imresize


class RandomRotate(object):
    def __call__(self, image, label):
        k = random.randint(0, 3)
        image = np.rot90(image, k)

        return image, label


class RandomMirror(object):
    def __call__(self, image, label):
        if random.randint(0, 2):
            image = image[:, ::-1]

        return image, label


class Resize(object):
    def __init__(self, size=224):
        self.size = size

    def __call__(self, image, label):
        image = imresize(image, (self.size, self.size))

        return image, label


class Normaliza(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, label):
        image = (image - self.mean) / self.std

        return image.astype(np.float32), label


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label):
        for t in self.transforms:
            img, label = t(img, label)

        return img, label


class Augmentaton(object):
    def __init__(self, size=224, mean=65.976, std=49.540):
        self.mean = mean
        self.std = std
        self.size = size
        self.augment = Compose([
            RandomMirror(),
            RandomRotate(),
            Resize(self.size),
            Normaliza(self.mean, self.std)
        ])

    def __call__(self, image, label):
        return self.augment(image, label)
