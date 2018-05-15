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
import cv2

import tensorflow as tf
from tensorflow.python import debug as tf_debug
# import keras.backend.tensorflow_backend as K
from tensorflow.python.keras._impl.keras import backend as K

import utils.utils as utils
import utils.utils_image as utils_image
from utils.config import process_config


class TrainerDensenet(BaseTrain):

    def __init__(self, sess, model, data, config,logger):
        super(TrainerDensenet, self).__init__(sess, model, data, config,logger)

        ## Configure estimator
        est_config =  tf.estimator.RunConfig(
                                    # model_dir=None,                                               # None: Use a default value set by the Estimator.
                                    # tf_random_seed=None,
                                    # save_checkpoints_secs=_USE_DEFAULT,                           # Default: 600 sec
                                    # keep_checkpoint_every_n_hours=10000,
                                    # log_step_count_steps=100,                                     # The frequency, in number of global steps, that the global step/sec and the loss will be logged during training.
                                    # train_distribute=None,
                                    # session_config=sess_config,
                                    keep_checkpoint_max=self.config.train_keep_checkpoint_max,
                                    save_checkpoints_steps=self.config.train_save_checkpoints_steps,# save_checkpoints_secs OR save_checkpoints_steps
                                    save_summary_steps=self.config.train_save_summary_steps         # default:100
                                    )

        params = {"learning_rate": self.config.learning_rate}
        model_dir = os.path.join(os.getcwd(), self.config.checkpoint_dir, 'keras')


        logging.debug('====================')
        # logging.debug(self.model.model.layers[0].output())
        # logging.debug('00000000000000000000000000000000{}'.format(self.model.model.layers[0]))

        ## Create Estimator from Keras model
        self.estimator = tf.keras.estimator.model_to_estimator(
                                                        keras_model=self.model.model,               # Keras model in memory.
                                                        keras_model_path=None,                      # Directory to a keras model on disk.
                                                        config=est_config,
                                                        custom_objects=params,
                                                        model_dir=model_dir                         # Directory to save Estimator model parameters, graph and etc.
                                                        )

        # self.estimator = tf.estimator.Estimator(
        #                         model_fn=self.model.model_densenet121_tf,
        #                         params=params,
        #                         config=est_config,
        #                         model_dir=model_dir,
        #                         warm_start_from=None
        #                         )


    def train_tf(self):

        self.estimator.train(
                       input_fn=lambda: self.data.input_fn(
                            file_pattern=os.path.join(self.config.tfrecords_path_train, '*.tfr'),
                            train=True,
                            batch_size=self.config.batch_size,
                            buffer_size=self.config.data_gen_buffer_size
                            ),
                        steps=2000)


    def train_and_eval(self):

        train_spec = tf.estimator.TrainSpec(
                                    input_fn=lambda: self.data.input_fn(
                                        file_pattern=os.path.join(self.config.tfrecords_path_train, '*.tfr'),
                                        train=True,
                                        batch_size=self.config.batch_size,
                                        buffer_size=self.config.data_gen_buffer_size
                                        ),
                                    max_steps=self.config.train_max_steps,                          # If None, train forever. The training input_fn is not expected to generate OutOfRangeError or StopIteration exceptions.
                                    hooks=None,
                                    )

        eval_spec = tf.estimator.EvalSpec(
                                    input_fn=lambda: self.data.input_fn(
                                        file_pattern=os.path.join(self.config.tfrecords_path_val, '*.tfr'),
                                        train=False,
                                        batch_size=self.config.batch_size_eval,
                                        buffer_size=self.config.data_gen_buffer_size
                                        ),
                                    steps=None,                                                     # default:100. No of steps for which to evaluate model. If None, evaluates until input_fn raises an end-of-input exception
                                    # start_delay_secs=120,                                           # default:120. Start evaluating after waiting for this many seconds.
                                    # throttle_secs=180,                                              # default:600. Evaluate after throttle_secs
                                    start_delay_secs=10,
                                    throttle_secs=10,
                                    name='eval_val'
                                    )

        tf.estimator.train_and_evaluate(self.estimator, train_spec, eval_spec)


    def train(self):
        hooks = []
        if self.config.debug == 1:
            hooks = [tf_debug.LocalCLIDebugHook()]

        logging.debug('\n=========================\nTRAIN')
        self.estimator.train(
                        input_fn=lambda: self.data.input_fn(
                            file_pattern=os.path.join(self.config.tfrecords_path_train, '*.tfr'),
                            train=True,
                            batch_size=self.config.batch_size,
                            buffer_size=self.config.data_gen_buffer_size
                            ),
                        max_steps=self.config.train_max_steps,
                        hooks=hooks,
                        # steps=None,                                                               # steps OR max_steps
                        # saving_listeners=None                                                     # Used for callbacks that run immediately before or after checkpoint savings.
                        )
        logging.debug('\n')


    def evaluate(self):

        ## Evaluate on train dataset
        logging.debug('\n=========================\nEVAL [dataset=train]')
        self.estimator.evaluate(
                        input_fn=lambda: self.data.input_fn(
                            file_pattern=os.path.join(self.config.tfrecords_path_train, '*.tfr'),
                            train=False,
                            batch_size=self.config.batch_size_eval,
                            buffer_size=self.config.data_gen_buffer_size
                            ),
                        # steps=100,
                        steps=None,                                                                 # If None, evaluates until input_fn raises an end-of-input exception.
                        checkpoint_path=None,                                                       # If None, the latest checkpoint in model_dir is used.
                        # checkpoint_path='output/densenet/checkpoints/keras/model.ckpt-2501',
                        hooks=None,
                        name="train"
                        )


        ## Evaluate on validation dataset
        logging.debug('\n=========================\nEVAL [dataset=val]')
        self.estimator.evaluate(
                        input_fn=lambda: self.data.input_fn(
                            file_pattern=os.path.join(self.config.tfrecords_path_val, '*.tfr'),
                            train=False,
                            batch_size=self.config.batch_size_eval,
                            buffer_size=self.config.data_gen_buffer_size
                            ),
                        steps=None,
                        checkpoint_path=None,
                        hooks=None,
                        name="val"
                        )


        ## Evaluate on test dataset
        logging.debug('\n=========================\nEVAL [dataset=test]')
        self.estimator.evaluate(
                        input_fn=lambda: self.data.input_fn(
                            file_pattern=os.path.join(self.config.tfrecords_path_test, '*.tfr'),
                            train=False,
                            batch_size=self.config.batch_size_eval,
                            buffer_size=self.config.data_gen_buffer_size
                            ),
                        steps=None,
                        checkpoint_path=None,
                        hooks=None,
                        name="test"
                        )
        logging.debug('\n')


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
        image_paths = utils_image.get_images_path_list_from_dir(self.config.tfrecords_path_train, img_format='jpg')
        # image_paths = utils_image.get_images_path_list_from_dir(self.config.tfrecords_path_test, img_format='jpg')
        # image_paths = utils_image.get_images_path_list_from_dir(self.config.tfrecords_path_val, img_format='jpg')

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

            # img = cv2.imread(image_paths[i])
            # img = cv2.resize(img, (224, 224))

            ## Preprocess images
            image = np.float32(np.array(image))
            image = self.data.preprocess_data(image)

            images.append(image)

        images = np.array(images)
        logging.debug('model_name {}'.format(self.config.model_name))
        logging.debug('images {}'.format(images.shape))

        # TODO: Don't shuffle else labels will mismatch
        x_key = self.config.model_name + '_input'
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={x_key: images},
                y=None,
                batch_size=self.config.batch_size_pred,
                num_epochs=1,
                shuffle=False,
                queue_capacity=1000,
                # In order to have predicted and repeatable order of reading and enqueueing,
                # such as in prediction and evaluation mode, num_threads should be 1.
                num_threads=1)

        # with tf.Session() as sess:
        #     init_op = tf.global_variables_initializer()
        #     sess.run(init_op)

        #     # sess.run(self.model.model.output, feed_dict={self.model.model.input:images})
        #     # sess.run(self.model.model['class_prob'], feed_dict={self.model.model.input:images})
        #     # op = self.model.model.get_layer('class_prob')
        #     # op = self.model.model.get_layer('class_prob')
        #     # op = self.model.model.layers[0]
        #     op = self.model.model.get_layer('dense_1')
        #     out = sess.run(op, 
        #             feed_dict={self.model.model.input:images})
        #     logging.debug('out {}'.format(out))
        # exit(0)


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


