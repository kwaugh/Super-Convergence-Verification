import os
import scipy.io
import numpy as np
import utils
import pickle
import itertools
from PIL import Image

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# for imagenet
# input_shape = [224, 224, 3]
# for cifar
input_shape = [32, 32, 3]

checkpoint_steps = 1000
save_steps = 1000
num_steps = 10000

batch_size = 256

model_name = 'ResNet56'
log_dir = 'log/'

data_dir = '../data/cifar-10-batches-py'

num_classes = 10
data_batch_names = ['data_batch_{0}'.format(i) for i in range(1, 6)]
# A len(data_batch_names)x10000x3072 numpy array of uint8s.
# Each row of the array stores a 32x32 colour image.
# The first 1024 entries contain the red channel values, the next 1024 the
# green, and the final 1024 the blue.
# The image is stored in row-major order, so that the first 32 entries of the
# array are the red channel values of the first row of the image.
train_data = []
valid_data = []
# A list of len(data_batch_names)x10000 numbers in the range 0-9.
# The number at index i indicates the label of the ith image in the array data.
train_labels = []
valid_labels = []
for i in range(len(data_batch_names)):
    data_batches = unpickle(os.path.join(data_dir, data_batch_names[i]))
    train_data.append(data_batches[b'data'])
    train_labels.append(data_batches[b'labels'])

valid_data_batch = unpickle(os.path.join(data_dir, 'test_batch'))
valid_data.append(valid_data_batch[b'data'])
valid_labels.append(valid_data_batch[b'labels'])

# flatten them
train_data = list(itertools.chain.from_iterable(train_data))
train_labels = list(itertools.chain.from_iterable(train_labels))
valid_data = list(itertools.chain.from_iterable(valid_data))
valid_labels = list(itertools.chain.from_iterable(valid_labels))

# CLR variables
cycle_size = 10000
step_size = cycle_size // 2
min_lr = 0.1
max_lr = 3.0
