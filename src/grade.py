from __future__ import print_function
from os import path
from math import sin, cos, atan2, sqrt, pi
from run import *
from util import *

def evaluate(Predictor):
    valid_ims, valid_labels = listIms(False, True, False)
    predictor = Predictor()
    predictor.DATASET_TYPE = 'cifar'

    # overall stats
    accuracy = 0.0
    count = 0

    # keep track of accuracy per class
    # class_to_accuracy = {i: 0 for i in range(config.num_classes)}
    # class_to_count    = {i: 0 for i in range(config.num_classes)}

    import time

    now = time.time()

    for i in range(len(valid_ims)):
        im = valid_ims[i]
        label = valid_labels[i]
        pred = predictor.predict(im)
        truth = label

        # overall accuracy
        accuracy += int(pred == truth)
        count += 1

        # per class accuracy
        # class_to_accuracy[truth_year] += int(pred_year == truth_year)
        # class_to_count[truth_year] += 1

    accuracy /= count
    print('accuracy: %0.3f using %d samples in %0.3f sec.'
            % (accuracy, count, time.time()-now))

    # for i in range(config.num_classes):
    #     if class_to_count[i] != 0:
    #         class_to_accuracy[i] /= class_to_count[i]
    #     print('%d: %0.3f using %d samples.'
    #             % (i, class_to_accuracy[i], class_to_count[i]))

    # print()

    return accuracy

if __name__ == "__main__":
    evaluate(Predictor)
