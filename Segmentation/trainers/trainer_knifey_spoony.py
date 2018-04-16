##########################################################################################
# Training
##########################################################################################
from base.base_trainer import BaseTrain
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import os
import glob
import pudb

class TrainerKnifeySpoony(BaseTrain):
    def __init__(self, sess, model, data, config,logger):
        super(TrainerKnifeySpoony, self).__init__(sess, model, data, config,logger)


    def train(self):
        self.model.model_estimator.train(input_fn=self.train_input_fn, steps=self.config.train_num_steps)


    def train_input_fn(self):
        # files = tf.data.Dataset.list_files("/path/to/dataset/train-*.tfrecord")
        filenames_regex = os.path.join(self.config.tfrecords_path_train, '*.tfr')
        filenames = glob.glob(filenames_regex)
        if not filenames:
            print('ERROR: No .tfr files found')
            exit(1)
        return self.data.input_fn(filenames=filenames, train=True)


    def test_input_fn(self):
        filenames_regex = os.path.join(self.config.tfrecords_path_test, '*.tfr')
        filenames = glob.glob(filenames_regex)
        if not filenames:
            print('ERROR: No .tfr files found')
            exit(1)
        return self.data.input_fn(filenames=filenames, train=False)


