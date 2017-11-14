import os

import tensorflow as tf
import numpy as np

from PIL import Image

import config

_HEIGHT, _WIDTH, _DEPTH = config.input_shape
_NUM_CLASSES = config.num_classes

def load_image(path, input_shape):
    x = Image.open(path)
    x = x.convert('RGB')
    h, w, _ = input_shape
    x = x.resize([h, w])

    return x

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

def preprocess_image(image, is_training):
    """Preprocess a single image of layout [height, width, depth]."""
    if is_training:
        # Resize the image to add four extra pixels on each side.
        image = tf.image.resize_image_with_crop_or_pad(
                image, _HEIGHT + 8, _WIDTH + 8)

        # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
        image = tf.random_crop(image, [_HEIGHT, _WIDTH, _DEPTH])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

    # Subtract off the mean and divide by the variance of the pixels.
    image = tf.image.per_image_standardization(image)
    return image

# def parse_record(raw_record):
#   """Parse CIFAR-10 image and label from a raw record."""
#   # Every record consists of a label followed by the image, with a fixed number
#   # of bytes for each.
#   label_bytes = 1
#   image_bytes = _HEIGHT * _WIDTH * _DEPTH
#   record_bytes = label_bytes + image_bytes

#   # Convert bytes to a vector of uint8 that is record_bytes long.
#   record_vector = tf.decode_raw(raw_record, tf.uint8)

#   # The first byte represents the label, which we convert from uint8 to int32
#   # and then to one-hot.
#   label = tf.cast(record_vector[0], tf.int32)
#   label = tf.one_hot(label, _NUM_CLASSES)

#   # The remaining bytes after the label represent the image, which we reshape
#   # from [depth * height * width] to [depth, height, width].
#   depth_major = tf.reshape(
#       record_vector[label_bytes:record_bytes], [_DEPTH, _HEIGHT, _WIDTH])

#   # Convert from [depth, height, width] to [height, width, depth], and cast as
#   # float32.
#   image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

#   return image, label

# we need to reshape the images from a flattened 1024,1024,1024 array to
# 32x32x3
def get_samples(images, input_shape):
    w, h, c = input_shape
    num_ims = len(images)
    ims = np.zeros((num_ims, w, h, c))
    for i in range(num_ims):
        ims[i] = np.reshape(images[i], tuple(input_shape))
    return ims

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
        image = self.images[self.index]
        label = self.labels[self.index]
        self.index = (self.index + 1) % len(self.labels)

        return np.float32(preprocess_image(image, self.augment_data)), np.int32(label)
