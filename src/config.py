import os
import scipy.io
import numpy as np
import utils
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# for resnet
input_shape = [224, 224, 3]

checkpoint_steps = 1000
save_steps = 1000
num_steps = 800000

batch_size = 256

model_name = 'ResNet56'
log_dir = 'log/'

data_dir = '../data/cifar-10-batches-py/'
data_batch_names = ['data_batch_{0}'.format(i) for i in range(1, 6)]
# A len(data_batch_names)x10000x3072 numpy array of uint8s.
# Each row of the array stores a 32x32 colour image.
# The first 1024 entries contain the red channel values, the next 1024 the
# green, and the final 1024 the blue.
# The image is stored in row-major order, so that the first 32 entries of the
# array are the red channel values of the first row of the image.
data = []
# A list of len(data_batch_names)x10000 numbers in the range 0-9.
# The number at index i indicates the label of the ith image in the array data.
labels = []
for i in range(len(data_batch_names)):
    data_batches = unpickle(os.path.join(data_dir, data_batch_names[i]))
    data.append(data_batches[b'data'])
    labels.append(data_batches[b'labels'])
# data_batches = list(map(lambda name: unpickle(os.path.join(data_dir, name)),
        # data_batch_names))
