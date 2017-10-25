import os
import scipy.io
import numpy as np
import utils

# for vgg
input_shape = [128, 128, 3]
# for alexnet
# input_shape = [224, 224, 3]

checkpoint_steps = 1000
save_steps = 1000
num_steps = 500000

batch_size = 128

model_name = 'SunNet'
log_dir = 'log/'

data_dir = '/projects/cs381V.grauman/' # this is a symlink
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

num_classes = 25
