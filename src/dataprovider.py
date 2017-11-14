import os

import tensorflow as tf
import numpy as np

from PIL import Image

import config

_HEIGHT, _WIDTH, _DEPTH = config.input_shape
_NUM_CLASSES = config.num_classes

def record_dataset(filenames):
  """Returns an input pipeline Dataset from `filenames`."""
  record_bytes = _HEIGHT * _WIDTH * _DEPTH + 1
  return tf.contrib.data.FixedLengthRecordDataset(filenames, record_bytes)

_NUM_DATA_FILES = 5

def get_filenames(is_training, data_dir):
    """Returns a list of filenames."""
    data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')

    assert os.path.exists(data_dir), (
            'Run cifar10_download_and_extract.py first to download and extract the '
            'CIFAR-10 data.')

    if is_training:
        return [
                os.path.join(data_dir, 'data_batch_%d.bin' % i)
                for i in range(1, _NUM_DATA_FILES + 1)
                ]
    else:
        return [os.path.join(data_dir, 'test_batch.bin')]

def preprocess_image(x, input_shape, augment_data, crop_size=4):
    # if augment_data:
    #     print('augmenting data')
    #     print('input_shape:', input_shape)
    #     print('x:', x)
    #     bbox = np.array([0, 0] + input_shape[0:2])
    #     bbox += [1, 1, -1, -1] * np.random.randint(0, crop_size, 4)
    #     print('got bbox')

    #     x = x.crop(bbox)
    #     print('cropped')
    #     h, w, _ = input_shape
    #     x = x.resize([h, w])
    #     print('resized')

    # subtract off the mean and divide by the variance of the pixels
    x = np.float32(x)
    mean = np.mean(x)
    var = np.var(x)
    x = (x - mean) / var

    return x

def get_samples(images, input_shape):
    return list(map(lambda x : np.reshape(x, tuple(input_shape), 'F'), images))

class Datagen(object):
    def __init__(self, is_training, input_shape, num_classes, augment_data):
        self.is_training = is_training
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
    def __init__(self, is_training, input_shape, num_classes, augment_data):
        super().__init__(is_training, input_shape, num_classes, augment_data)

        if is_training:
            self.images = get_samples(config.train_data, input_shape)
            self.labels = config.train_labels
        else:
            self.images = get_samples(config.valid_data, input_shape)
            self.labels = config.valid_labels

        self.index = 0

        self.augment_data = augment_data
        self.input_shape = input_shape

        self._dtypes = [tf.float32, tf.int32]
        self._shapes = [input_shape, []]

    def __next__(self):
        label = self.labels[self.index]
        image = self.images[self.index]
        self.index = (self.index + 1) % len(self.labels)
        image = preprocess_image(image, self.input_shape, self.augment_data)

        return np.float32(image), np.int32(label)
