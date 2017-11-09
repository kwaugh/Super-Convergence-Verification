import os
import scipy.io
import numpy as np
import utils

# for resnet
input_shape = [224, 224, 3]

checkpoint_steps = 1000
save_steps = 1000
num_steps = 800000

batch_size = 256

model_name = 'ResNet56'
log_dir = 'log/'

data_dir = '../data/'
filenames = scipy.io.loadmat(os.path.join(data_dir, 'filenames.mat'))
classnames = filenames['classnames']
trainImNames = filenames['trainImNames']
test1ImNames = filenames['test1ImNames']

test2ImNames = scipy.io.loadmat(os.path.join(data_dir, 'test2ImNames.mat'))
test2ImNames = test2ImNames['test2ImNames']

extraTrainImNames = filenames['extraTrainImNames']
# concatenate trainImNames and extraTrainImNames
allTrainImNames = np.column_stack((trainImNames, extraTrainImNames))

class_to_label = utils.get_class_to_label()
label_to_class = utils.get_label_to_class()

num_classes = 10
