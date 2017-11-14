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
num_steps = 80000

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

# image = np.zeros((32, 32, 3))
# # stack the channels manually
# for d in range(3):
#     for r in range(32):
#         for c in range(32):
#             image[r, c, d] = valid_data[0][0][1024*d + 32*r + c]

# ims[i, :, :, 0] = np.reshape(images[i][0:1024], (32, 32)) # red
# ims[i, :, :, 1] = np.reshape(images[i][1024:2048], (32, 32)) # green
# ims[i, :, :, 2] = np.reshape(images[i][2048:], (32, 32)) # blue
# image = np.reshape(valid_data[0][0], tuple(input_shape), 'F')
# img = Image.fromarray(image, 'RGB')
# img.show()
# input('Press a key')

valid_labels.append(valid_data_batch[b'labels'])

# flatten them
train_data = list(itertools.chain.from_iterable(train_data))
train_labels = list(itertools.chain.from_iterable(train_labels))
valid_data = list(itertools.chain.from_iterable(valid_data))
valid_labels = list(itertools.chain.from_iterable(valid_labels))
