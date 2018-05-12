##########################################################################################
# Training
##########################################################################################
from base.base_trainer import BaseTrain

import os
import csv
import glob
import pudb
import random
import random
import logging
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import keras.backend.tensorflow_backend as K

import utils.utils as utils
import utils.utils_image as utils_image
from utils.config import process_config


class TrainerDensenet(BaseTrain):

    def __init__(self, sess, model, data, config,logger):
        super(TrainerDensenet, self).__init__(sess, model, data, config,logger)

        ## Configure estimator
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth=True
        sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4
        K.set_session(tf.Session(config=sess_config))

        est_config =  tf.estimator.RunConfig(
                                    # model_dir=None,                                               # None: Use a default value set by the Estimator.
                                    # tf_random_seed=None,
                                    # save_checkpoints_secs=_USE_DEFAULT,                           # Default: 600 sec
                                    # keep_checkpoint_every_n_hours=10000,
                                    # log_step_count_steps=100,                                     # The frequency, in number of global steps, that the global step/sec and the loss will be logged during training.
                                    # train_distribute=None,
                                    session_config=sess_config,
                                    keep_checkpoint_max=self.config.train_keep_checkpoint_max,
                                    save_checkpoints_steps=self.config.train_save_checkpoints_steps,# save_checkpoints_secs OR save_checkpoints_steps
                                    save_summary_steps=self.config.train_save_summary_steps         # default:100
                                    )

        params = {"learning_rate": self.config.learning_rate}
        model_dir = os.path.join(os.getcwd(), self.config.checkpoint_dir, 'keras')


        ## Create Estimator from Keras model
        self.estimator = tf.keras.estimator.model_to_estimator(
                                                                keras_model=self.model.model,       # Keras model in memory.
                                                                keras_model_path=None,              # Directory to a keras model on disk.
                                                                config=est_config,
                                                                custom_objects=params,
                                                                model_dir=model_dir                 # Directory to save Estimator model parameters, graph and etc.
                                                                )


    def train_and_eval(self):
        filenames_train = utils.get_files_from_pattern(os.path.join(self.config.tfrecords_path_train, '*.tfr'))
        filenames_val = utils.get_files_from_pattern(os.path.join(self.config.tfrecords_path_val, '*.tfr'))
        filenames_test = utils.get_files_from_pattern(os.path.join(self.config.tfrecords_path_val, '*.tfr'))

        train_spec = tf.estimator.TrainSpec(
                                        input_fn=lambda: self.data.input_fn(
                                            filenames=filenames_train,
                                            train=True,
                                            batch_size=self.config.batch_size,
                                            buffer_size=self.config.data_gen_buffer_size
                                            ),
                                        max_steps=self.config.train_max_steps,                      # If None, train forever. The training input_fn is not expected to generate OutOfRangeError or StopIteration exceptions.
                                        hooks=None,
                                        )

        eval_spec = tf.estimator.EvalSpec(
                                        input_fn=lambda: self.data.input_fn(
                                            filenames=filenames_val,
                                            train=False,
                                            batch_size=self.config.batch_size,
                                            buffer_size=self.config.data_gen_buffer_size
                                            ),
                                        steps=None,                                                 # default:100. No of steps for which to evaluate model. If None, evaluates until input_fn raises an end-of-input exception
                                        start_delay_secs=120,                                       # default:120. Start evaluating after waiting for this many seconds.
                                        throttle_secs=180,                                          # default:600. Evaluate after throttle_secs
                                        name='eval_val'
                                        )

        tf.estimator.train_and_evaluate(self.estimator, train_spec, eval_spec)


    def train(self):
        hooks = []
        if self.config.debug == 1:
            hooks = [tf_debug.LocalCLIDebugHook()]

        logging.debug('\n=========================\nTRAIN')
        self.estimator.train(
                        input_fn=lambda: self.train_input_fn(), 
                        hooks=hooks,
                        max_steps=self.config.train_max_steps,
                        # steps=None,                                                               # steps OR max_steps
                        # saving_listeners=None                                                     # Used for callbacks that run immediately before or after checkpoint savings.
                        )
        logging.debug('\n')

    def train_input_fn(self):
        filenames_train = utils.get_files_from_pattern(os.path.join(self.config.tfrecords_path_train, '*.tfr'))
        return self.data.input_fn(filenames=filenames_train, train=True, batch_size=self.config.batch_size, buffer_size=self.config.data_gen_buffer_size)


    def evaluate(self):
        logging.debug('\n=========================\nEVAL [dataset=train]')
        self.estimator.evaluate(
                            input_fn=lambda: self.eval_input_fn(self.config.tfrecords_path_train), 
                            steps=None,                                                             # If None, evaluates until input_fn raises an end-of-input exception.
                            hooks=None,
                            checkpoint_path=None,                                                   # If None, the latest checkpoint in model_dir is used.
                            name="eval_train")

        logging.debug('\n=========================\nEVAL [dataset=val]')
        self.estimator.evaluate(
                            input_fn=lambda: self.eval_input_fn(self.config.tfrecords_path_val), 
                            steps=None,
                            hooks=None,
                            checkpoint_path=None,
                            name="eval_val")

        logging.debug('\n=========================\nEVAL [dataset=test]')
        self.estimator.evaluate(
                            input_fn=lambda: self.eval_input_fn(self.config.tfrecords_path_test), 
                            steps=None,
                            hooks=None,
                            checkpoint_path=None,
                            name="eval_test")
        logging.debug('\n')


    def eval_input_fn(self, tfrecords_path):
        filenames_train = utils.get_files_from_pattern(os.path.join(self.config.tfrecords_path, '*.tfr'))
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
        # TODO: Check predict_keys
        predictions = self.estimator.predict(
                                        input_fn=predict_input_fn,
                                        checkpoint_path=checkpoint_path,
                                        hooks=None,
                                        predict_keys=None,
                                        yield_single_examples=True
                                        )

        class_prob = [p['class_prob'] for p in predictions]
        # class_prob = [p['dense_4'] for p in predictions]
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





# CODE for future use
"""
hooks=[
    tf.train.LoggingTensorHook(["layer_conv1/bias/Adam:0"], every_n_iter=1),
    tf.train.LoggingTensorHook(["layer_conv1/bias/Adam:0"], every_n_iter=1),
    tf.train.LoggingTensorHook(["Reshape:0:DebugIdentity"], every_n_iter=1),
    tf.train.CheckpointSaverHook("./tmp", save_steps=1000, checkpoint_basename='model.ckpt'),
    tf.train.StepCounterHook(every_n_steps=100, output_dir="./tmp"),
    tf.train.SummarySaverHook(save_steps=100, output_dir="./tmp"),
]
"""


