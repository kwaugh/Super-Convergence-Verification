import argparse
import csv
import os

from os import path

import numpy as np

from PIL import Image

from util import *
import model
import config


def load(image_path, input_shape):
    return np.float32(Image.open(image_path).convert('RGB').resize(input_shape[:-1]))


class Predictor:
    sun_network = None
    DATASET_TYPE = 'sun'

    def predict(self, image_path):
        return self.predict_sun(image_path)

    def predict_sun(self, image_path):
        if Predictor.sun_network is None:
            save_path = os.path.join(config.log_dir, config.model_name)
            num_classes = config.num_classes

            Predictor.sun_network = model.SunNetRunner(
                    save_path, num_classes)

        image = load(image_path, config.input_shape)

        return Predictor.sun_network.predict(image)
