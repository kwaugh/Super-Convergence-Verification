import os

import tensorflow as tf
import numpy as np

from PIL import Image

import config

def load_image(path, input_shape):
    x = Image.open(path)
    x = x.convert('RGB')
    h, w, _ = input_shape
    x = x.resize([h, w])

    return x

def augment_data(x, input_shape, theta=5, crop_size=10):
    x = x.rotate(np.random.uniform(low=-theta, high=theta))

    bbox = np.array([0, 0] + list(x.size))
    bbox += [1, 1, -1, -1] * np.random.randint(0, crop_size, 4)

    x = x.crop(bbox)
    h, w, _ = input_shape
    x = x.resize([h, w])
    x = np.float32(x)

    return x

def get_samples(train_or_test, input_shape):
    image_path_label = []

    for i in range(len(train_or_test)):
        for j in range(len(train_or_test[i])):
            # we need to only grab the ones that are real
            if isinstance(train_or_test[i][j][0], str):
                image = load_image(train_or_test[i][j][0], input_shape)
                image_path_label.append((image, i))
                # image_path = train_or_test[i][j][0]
                # image_path_label.append((image_path, i))

    return image_path_label

class Datagen(object):
    def __init__(self, train_or_test, input_shape, num_classes, augment_data):
        self.train_or_test = train_or_test

        self.input_shape = input_shape
        self.num_classes = num_classes

        self._dtypes = None
        self._shapes = None

    def __next__(self):
        raise NotImplementedError('__next__() has not been overloaded.')

    def __iter__(self):
        return self

    def get_dtypes(self):
        return self._dtypes

    def get_shapes(self):
        return self._shapes


class SparseDatagen(Datagen):
    """
    This generator yields class labels as ints.

    Example:
        (image_path_0, 3)
        (image_path_1, 25)
        ...
    """
    def __init__(self, train_or_test, input_shape, num_classes, augment_data):
        super().__init__(train_or_test, input_shape, num_classes, augment_data)

        self.image_path_label = get_samples(train_or_test, input_shape)
        self.augment_data = augment_data
        self.input_shape = input_shape

        self.index = 0

        self._dtypes = [tf.float32, tf.int32]
        self._shapes = [input_shape, []]

    def __next__(self):
        image, label = self.image_path_label[self.index]
        # image_path, label = self.image_path_label[self.index]
        # image = load_image(image_path, self.input_shape)

        self.index = (self.index + 1) % len(self.image_path_label)

        if self.augment_data:
            return augment_data(image, self.input_shape), np.int32(label)
        else:
            return np.float32(image), np.int32(label)
