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
import logging
from PIL import Image, ImageOps
import csv
import random

from data_handler.tfrecords_densenet import TFRecordsDensenet
import utils.utils as utils
import utils.utils_image as utils_image
from utils.config import process_config


class TrainerDensenet(BaseTrain):
    def __init__(self, sess, model, data, config,logger):
        super(TrainerDensenet, self).__init__(sess, model, data, config,logger)


    def train_and_eval(self):

        filenames_regex = os.path.join(self.config.tfrecords_path_train, '*.tfr')
        filenames_train = glob.glob(filenames_regex)
        if not filenames_train:
            logging.error('ERROR: No .tfr files found')
            exit(1)
        logging.debug('filenames_train {}'.format(filenames_train))

        filenames_regex = os.path.join(self.config.tfrecords_path_test, '*.tfr')
        filenames_test = glob.glob(filenames_regex)
        if not filenames_test:
            logging.error('ERROR: No .tfr files found')
            exit(1)
        logging.debug('filenames_test {}'.format(filenames_test))


        train_spec = tf.estimator.TrainSpec(input_fn=lambda: self.data.input_fn(
                                                                    filenames=filenames_train, 
                                                                    train=True, 
                                                                    batch_size=self.config.batch_size, 
                                                                    buffer_size=self.config.data_gen_buffer_size),
                                                                    max_steps=self.config.train_max_steps,
                                                                    )

        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: self.data.input_fn(
                                                                    filenames=filenames_test, 
                                                                    train=False, 
                                                                    batch_size=self.config.batch_size, 
                                                                    buffer_size=self.config.data_gen_buffer_size)
                                                                    )

        tf.estimator.train_and_evaluate(self.model.model_estimator, train_spec, eval_spec)


    def train():

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

        logging.debug('\n=========================')
        logging.debug('TRAIN')
        self.model.model_estimator.train(input_fn=lambda: self.train_input_fn(), steps=self.config.train_max_steps, hooks=hooks)
        logging.debug('\n')


    def train_input_fn(self):
        # files = tf.data.Dataset.list_files("/path/to/dataset/train-*.tfrecord")
        filenames_regex = os.path.join(self.config.tfrecords_path_train, '*.tfr')
        filenames = glob.glob(filenames_regex)
        if not filenames:
            logging.debug('ERROR: No .tfr files found')
            exit(1)
        return self.data.input_fn(filenames=filenames, train=True, batch_size=self.config.batch_size, buffer_size=self.config.data_gen_buffer_size)


    def evaluate(self):
        """
        Estimator.evaluate(input_fn, steps=None, hooks=None, checkpoint_path=None, name=None)
        """
        logging.debug('\n=========================')
        logging.debug('EVAL [dataset=train]')
        self.model.model_estimator.evaluate(input_fn=lambda: self.eval_input_fn(self.config.tfrecords_path_train), name="train")

        logging.debug('\n=========================')
        logging.debug('EVAL [dataset=val]')
        self.model.model_estimator.evaluate(input_fn=lambda: self.eval_input_fn(self.config.tfrecords_path_val), name="val")
        logging.debug('\n')

        logging.debug('\n=========================')
        logging.debug('EVAL [dataset=test]')
        self.model.model_estimator.evaluate(input_fn=lambda: self.eval_input_fn(self.config.tfrecords_path_test), name="test")
        logging.debug('\n')


    def eval_input_fn(self, tfrecords_path):
        filenames_regex = os.path.join(tfrecords_path, '*.tfr')
        filenames = glob.glob(filenames_regex)
        if not filenames:
            logging.error('ERROR: No .tfr files found')
            exit(1)
        return self.data.input_fn(filenames=filenames, train=False, batch_size=self.config.batch_size, buffer_size=self.config.data_gen_buffer_size)


    def predict(self):
        ## Get image-label mapping
        image_label_dict = {}
        dataset_labels_file_path = 'datasets/densenet/ISIC2018_Task3_Training_GroundTruth.csv'
        with open(dataset_labels_file_path) as csvfile:
            read_csv = csv.reader(csvfile, delimiter=',')
            for index, row in enumerate(read_csv):
                ## Skip header
                if index == 0:
                    continue
                label_one_hot_encoding = [int(round(float(row[i+1]), 0)) for i in range(7)]
                image_label_dict[row[0]] = np.argmax(label_one_hot_encoding)


        ## Get image paths
        # filenames_regex = os.path.join(self.config.tfrecords_path_val, '*.jpg')
        filenames_regex = os.path.join(self.config.tfrecords_path_test, '*.jpg')
        image_paths = glob.glob(filenames_regex)
        if not image_paths:
            logging.error('ERROR: No images found')
            exit(1)

        ## Sample n images
        random.shuffle(image_paths)
        image_paths = image_paths[0:self.config.predict_num_images]


        ## Get gt_labels
        gt_labels = []
        for image_path in image_paths:
            image_name = os.path.basename(image_path).rsplit('.', 1)[0]
            gt_labels.append(image_label_dict[image_name])


        images = []
        for image_path in image_paths:

            ## Load image
            image = Image.open(image_path)

            ## Resize and center crop image. size: (width, height)
            image = ImageOps.fit(image, (self.config.tfr_image_width, self.config.tfr_image_height), Image.LANCZOS, 0, (0.5, 0.5))

            ## Preprocess images
            image = np.float32(np.array(image))
            image = self.data.preprocess_data(image)

            images.append(image)

        images = np.array(images)

        # TODO: Don't shuffle else labels will mismatch
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"densenet121_input": images},
                y=None,
                batch_size=128,
                num_epochs=1,
                shuffle=False,
                queue_capacity=1000,
                # In order to have predicted and repeatable order of reading and enqueueing,
                # such as in prediction and evaluation mode, num_threads should be 1.
                num_threads=1)


        checkpoint_path = None
        if not self.config.predict_weights_path:
            checkpoint_path = self.config.predict_weights_path

        # NOTE: predictions is <generator object Estimator.predict> and hence (maybe) we can dereference it only once.
        predictions = self.model.model_estimator.predict(input_fn=predict_input_fn,
                                                         checkpoint_path=checkpoint_path)

        # class_prob = [p['class_prob'] for p in predictions]
        class_prob = [p['dense_4'] for p in predictions]
        pred_labels = np.argmax(np.array(class_prob), axis=1)

        for gt_label, pred_label in zip(gt_labels, pred_labels):
            print('GT, PRED: [{}, {}]'.format(gt_label, pred_label))


        ## Confusion matrix
        # https://stackoverflow.com/questions/41617463/tensorflow-confusion-matrix-in-tensorboard
        confusion = tf.confusion_matrix(labels=gt_labels, predictions=pred_labels, num_classes=self.config.num_classes)
        logging.debug('Row(GT), Col(Pred)')
        with tf.Session() as sess:
            print(sess.run(confusion))

        # Plot and save confusion matrix
        utils.get_confusion_matrix(self.config, gt_labels, pred_labels)

