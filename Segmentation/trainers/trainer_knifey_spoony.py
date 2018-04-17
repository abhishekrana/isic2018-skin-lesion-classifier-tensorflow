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
import random

import utils.utils_image as utils_image
from data_handler.tfrecords_knifey_spoony import TFRecordsKnifeySpoony


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


    def evaluate(self):
        self.model.model_estimator.evaluate(input_fn=self.test_input_fn)

        # name: to run multiple evaluations on different data sets, such as on training data vs test data
        # evaluate(input_fn, steps=None, hooks=None, checkpoint_path=None, name=None)


    def test_input_fn(self):
        filenames_regex = os.path.join(self.config.tfrecords_path_test, '*.tfr')
        filenames = glob.glob(filenames_regex)
        if not filenames:
            print('ERROR: No .tfr files found')
            exit(1)
        return self.data.input_fn(filenames=filenames, train=False)


    def predict(self):

        # image_paths_list, gt_labels = self.data.read_dataset(self.config.dataset_path_train)
        image_paths_list, gt_labels = self.data.read_dataset(self.config.dataset_path_test)
        print('image_paths_list', image_paths_list[0])

        image_idx_rand = random.sample(range(1, len(image_paths_list)), 10)

        image_paths_list = np.array(image_paths_list)[image_idx_rand].tolist()
        gt_labels = np.array(gt_labels)[image_idx_rand].tolist()

        image = utils_image.load_images(image_paths=image_paths_list)

        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"image": image.astype(np.float32)},
                num_epochs=1,
                shuffle=False)

        predictions = self.model.model_estimator.predict(input_fn=predict_input_fn)
        cls_pred = np.array(list(predictions))

        print('gt_labels', gt_labels)
        print('cls_pred', cls_pred)



