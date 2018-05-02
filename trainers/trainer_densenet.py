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
from PIL import Image

import utils.utils_image as utils_image
from data_handler.tfrecords_densenet import TFRecordsDensenet


class TrainerDensenet(BaseTrain):
    def __init__(self, sess, model, data, config,logger):
        super(TrainerDensenet, self).__init__(sess, model, data, config,logger)


    def train(self):

        # labels_categorical = tf.keras.utils.to_categorical(labels, self.config.num_classes)

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
                                                                    max_steps=2000
                                                                    )


        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: self.data.input_fn(
                                                                    filenames=filenames_test, 
                                                                    train=False, 
                                                                    batch_size=self.config.batch_size, 
                                                                    buffer_size=self.config.data_gen_buffer_size)
                                                                    )
         

        tf.estimator.train_and_evaluate(self.model.model_estimator, train_spec, eval_spec)

        # eval_spec = tf.estimator.EvalSpec(input_fn=self.eval_input_fn)
        # train_spec = tf.estimator.TrainSpec(input_fn=self.train_input_fn)

        # hooks=[
        # tf.train.LoggingTensorHook(["layer_conv1/bias/Adam:0"], every_n_iter=1),
        # tf.train.LoggingTensorHook(["Reshape:0:DebugIdentity"], every_n_iter=1),
        # tf.train.CheckpointSaverHook("./tmp", save_steps=1000, checkpoint_basename='model.ckpt'),
        # tf.train.StepCounterHook(every_n_steps=100, output_dir="./tmp"),
        # tf.train.SummarySaverHook(save_steps=100, output_dir="./tmp"),
        # ]

        # hooks = []
        # if self.config.debug == 1:
        #     hooks = [tf_debug.LocalCLIDebugHook()]
        #     # hooks=[tf.train.LoggingTensorHook(["layer_conv1/bias/Adam:0"], every_n_iter=1)]

        # logging.debug('\n=========================')
        # logging.debug('TRAIN')
        # self.model.model_estimator.train(input_fn=lambda: self.train_input_fn(), steps=self.config.train_num_steps, hooks=hooks)
        # logging.debug('\n')




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


        test_files = []
        for label in self.config.labels:
            test_label = [os.path.join(os.path.join(self.config.dataset_path_test, label), file_name) for file_name in os.listdir(os.path.join(self.config.dataset_path_test, label))]
            test_files += test_label
    
        # test_dogs = [os.path.join(os.path.join(self.config.dataset_path_test, 'dogs'), file_name) for file_name in os.listdir(os.path.join(self.config.dataset_path_test, 'dogs'))]
        # test_cats = [os.path.join(os.path.join(self.config.dataset_path_test, 'cats'), file_name) for file_name in os.listdir(os.path.join(self.config.dataset_path_test, 'cats'))]
        # test_dogs = [os.path.join(os.path.join(self.config.dataset_path_test, 'dogs'), file_name) for file_name in os.listdir(os.path.join(self.config.dataset_path_test, 'dogs'))]
        # test_files = test_cats + test_dogs

        predict_results = self.model.model_estimator.predict(input_fn=lambda: self.predict_imgs_input_fn(test_files[:10], batch_size=10))

        predict_logits = []
        for prediction in predict_results:
            predict_logits.append(prediction['dense_2'][0])


        predict_is_dog = [logit > 0.5 for logit in predict_logits[:10]]
        actual_is_dog = [label > 0.5 for label in test_labels[:10]]
        print("Predict dog:",predict_is_dog)
        print("Actual dog :",actual_is_dog)


        return 


        no_images_predict = 20

        # image_paths_list, gt_labels = self.data.read_dataset(self.config.dataset_path_train)
        image_paths_list, gt_labels = self.data.read_dataset(self.config.dataset_path_test)
        logging.debug('image_paths_list: {}'.format(len(image_paths_list)))

        image_idx_rand = random.sample(range(1, len(image_paths_list)), no_images_predict)
        logging.debug('image_idx_rand :{}', image_idx_rand)

        image_paths_list = np.array(image_paths_list)[image_idx_rand].tolist()
        gt_labels = np.array(gt_labels)[image_idx_rand].tolist()

        for i in range(no_images_predict):
            logging.debug('image_paths_list: {},  label:{}'.format(image_paths_list[i], gt_labels[i]))

        # Load batch of images in float32 format from list of image names
        images = np.array([np.array(Image.open(image_path), dtype=np.float32) for image_path in image_paths_list])

        # TODO: Maybe change from RGB to BGR format as saved weights have this format

        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"image": images},
                num_epochs=1,
                shuffle=False)

        logging.debug('\n=========================')
        logging.debug('PREDICT')
        predictions = self.model.model_estimator.predict(input_fn=predict_input_fn)
        logging.debug('\n')

        cls_pred = np.array(list(predictions))
        logging.debug(len(cls_pred))

        for i in range(no_images_predict):
            logging.debug('[GT Pred] [{} {}]'.format(gt_labels[i], cls_pred[i]))


    def predict_imgs_input_fn(self, filenames, batch_size=1):
        pu.db
        def _parse_function(filename, label):
            image_string = tf.read_file(filename)
            image = tf.image.decode_image(image_string, channels=3)
            image.set_shape([None, None, None])
            image = tf.image.resize_images(image, [self.config.tfr_image_height, self.config.tfr_image_width])
            image = tf.subtract(image, 116.779) # Zero-center by mean pixel
            image.set_shape([self.config.tfr_image_height, self.config.tfr_image_width, self.config.tfr_image_channels])
            image = tf.reverse(image, axis=[2]) # 'RGB'->'BGR'
            # d = dict(zip([input_name], [image])), label
            d = image, label
            return d
        labels = [0]*len(filenames)
        labels=np.array(labels)
        # Expand the shape of "labels" if necessory
        if len(labels.shape) == 1:
            labels = np.expand_dims(labels, axis=1)
        filenames = tf.constant(filenames)
        labels = tf.constant(labels)
        labels = tf.cast(labels, tf.float32)
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        dataset = dataset.map(_parse_function)
        dataset = dataset.batch(batch_size)  # Batch size to use
        iterator = dataset.make_one_shot_iterator()
        batch_features, batch_labels = iterator.get_next()
        return batch_features, batch_labels
