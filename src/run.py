import argparse
import csv
import os

from os import path

import numpy as np

from PIL import Image

from util import *
import model
import config


# def load(image_path, input_shape):
#     return np.float32(Image.open(image_path).convert('RGB').resize(input_shape[:-1]))


class Predictor:
    network = None
    DATASET_TYPE = 'cifar'

    def predict(self, image):
        if Predictor.network is None:
            save_path = os.path.join(config.log_dir, config.model_name)
            num_classes = config.num_classes

            Predictor.network = model.NetRunner(
                    save_path, num_classes)

        return Predictor.network.predict(image)
