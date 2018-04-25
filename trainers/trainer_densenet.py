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
from tensorflow.python import debug as tf_debug
from PIL import Image

import utils.utils_image as utils_image
from data_handler.tfrecords_densenet import TFRecordsDensenet


class TrainerDensenet(BaseTrain):
    def __init__(self, sess, model, data, config,logger):
        super(TrainerDensenet, self).__init__(sess, model, data, config,logger)


    def train(self):

        # hooks=[
        # tf.train.LoggingTensorHook(["layer_conv1/bias/Adam:0"], every_n_iter=1),
        # tf.train.LoggingTensorHook(["Reshape:0:DebugIdentity"], every_n_iter=1),
        # tf.train.CheckpointSaverHook("./tmp", save_steps=1000, checkpoint_basename='model.ckpt'),
        # tf.train.StepCounterHook(every_n_steps=100, output_dir="./tmp"),
        # tf.train.SummarySaverHook(save_steps=100, output_dir="./tmp"),
        # ]

        hooks = []
        if self.config.debug == 1:
            hooks = [tf_debug.LocalCLIDebugHook()]
            # hooks=[tf.train.LoggingTensorHook(["layer_conv1/bias/Adam:0"], every_n_iter=1)]

        print('\n=========================')
        print('TRAIN')
        self.model.model_estimator.train(input_fn=self.train_input_fn, steps=self.config.train_num_steps, hooks=hooks)
        print('\n')


    def train_input_fn(self):
        # files = tf.data.Dataset.list_files("/path/to/dataset/train-*.tfrecord")
        filenames_regex = os.path.join(self.config.tfrecords_path_train, '*.tfr')
        filenames = glob.glob(filenames_regex)
        if not filenames:
            print('ERROR: No .tfr files found')
            exit(1)
        return self.data.input_fn(filenames=filenames, train=True)


    def evaluate(self):
        print('\n=========================')
        print('EVAL [dataset=train]')
        self.model.model_estimator.evaluate(input_fn=self.eval_train_input_fn, name="train")

        print('\n=========================')
        print('EVAL [dataset=test]')
        self.model.model_estimator.evaluate(input_fn=self.eval_test_input_fn, name="test")
        print('\n')

        # name: to run multiple evaluations on different data sets, such as on training data vs test data
        # evaluate(input_fn, steps=None, hooks=None, checkpoint_path=None, name=None)

    def eval_train_input_fn(self):
        filenames_regex = os.path.join(self.config.tfrecords_path_train, '*.tfr')
        filenames = glob.glob(filenames_regex)
        if not filenames:
            print('ERROR: No .tfr files found')
            exit(1)
        return self.data.input_fn(filenames=filenames, train=False)

    def eval_test_input_fn(self):
        filenames_regex = os.path.join(self.config.tfrecords_path_test, '*.tfr')
        filenames = glob.glob(filenames_regex)
        if not filenames:
            print('ERROR: No .tfr files found')
            exit(1)
        return self.data.input_fn(filenames=filenames, train=False)


    def predict(self):

        no_images_predict = 20

        # image_paths_list, gt_labels = self.data.read_dataset(self.config.dataset_path_train)
        image_paths_list, gt_labels = self.data.read_dataset(self.config.dataset_path_test)
        print('image_paths_list: {}'.format(len(image_paths_list)))

        image_idx_rand = random.sample(range(1, len(image_paths_list)), no_images_predict)
        print('image_idx_rand :{}', image_idx_rand)

        # pu.db
        image_paths_list = np.array(image_paths_list)[image_idx_rand].tolist()
        gt_labels = np.array(gt_labels)[image_idx_rand].tolist()

        for i in range(no_images_predict):
            print('image_paths_list: {},  label:{}'.format(image_paths_list[i], gt_labels[i]))

        # Load batch of images in float32 format from list of image names
        images = np.array([np.array(Image.open(image_path), dtype=np.float32) for image_path in image_paths_list])

        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"image": images},
                num_epochs=1,
                shuffle=False)

        print('\n=========================')
        print('PREDICT')
        predictions = self.model.model_estimator.predict(input_fn=predict_input_fn)
        print('\n')

        cls_pred = np.array(list(predictions))
        print(len(cls_pred))

        for i in range(no_images_predict):
            print('[GT Pred] [{} {}]'.format(gt_labels[i], cls_pred[i]))



