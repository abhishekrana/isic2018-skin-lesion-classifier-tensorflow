##########################################################################################
# Data Generator
##########################################################################################
import os
os.sys.path.append('./')

from base.base_data import BaseData
import numpy as np
import tensorflow as tf
import glob
import pudb
import logging
from bunch import Bunch

import utils.utils as utils
import utils.utils_image as utils_image
from utils.config import process_config

from tensorflow.python.keras._impl.keras.applications import imagenet_utils
from tensorflow.python.keras.preprocessing import image as k_image
from tensorflow.python.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
from tensorflow.python.keras.applications.xception import preprocess_input as preprocess_input_xception
from tensorflow.python.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
from tensorflow.python.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_input_inception_resnet_v2
from tensorflow.python.keras.applications.densenet import preprocess_input as preprocess_input_densenet


class DataGeneratorDensenet(BaseData):
    def __init__(self, config):
        super(DataGeneratorDensenet, self).__init__(config)


    def data_preprocessing(self, image):
        logging.debug('image {}'.format(image))


        image = tf.reshape(image, [self.config.tfr_image_height, self.config.tfr_image_width, self.config.tfr_image_channels])

        ## Densenet specific
        # TODO: _aSk As per test_inference.py, trained weights look like to be in BGR format. Verify this.
        ## Subtract mean pixel and multiple by scaling constant
        # https://github.com/flyyufelix/DenseNet-Keras/blob/master/test_inference.py
        # https://github.com/shicai/DenseNet-Caffe
        # Convert RGB to BGR
        # red, green, blue = tf.split(axis=2, num_or_size_splits=3, value=image)
        # assert red.get_shape().as_list() == [224, 224, 1]
        # assert green.get_shape().as_list() == [224, 224, 1]
        # assert blue.get_shape().as_list() == [224, 224, 1]
        # densenet_mean_b = 103.94
        # densenet_mean_g = 116.78
        # densenet_mean_r = 123.68
        # densenet_scale = 0.017
        # image_bgr = tf.concat(axis=2, values=[
        #     blue - densenet_mean_b,
        #     green - densenet_mean_g,
        #     red - densenet_mean_r,
	# ])
        # image_bgr_scaled = image_bgr * densenet_scale
        # return image_bgr_scaled

        return image


    def preprocess_data(self, image):
        """
        https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/python/keras/_impl/keras/applications/imagenet_utils.py

        mode:
          - caffe: will convert the images from RGB to BGR, then will zero-center each
                color channel with respect to the ImageNet dataset, without scaling.
          - tf: will scale pixels between -1 and 1, sample-wise.
          - torch: will scale pixels between 0 and 1 and then will normalize each channel
                with respect to the ImageNet dataset.


	@tf_export('keras.applications.resnet50.preprocess_input',
		   'keras.applications.vgg19.preprocess_input',
		   'keras.applications.vgg16.preprocess_input')
	def preprocess_input(x, data_format=None, mode='caffe'):
            x: Input Numpy or symbolic tensor, 3D or 4D.


        @tf_export('keras.applications.inception_resnet_v2.preprocess_input')
        def preprocess_input(x):
            x: a 4D numpy array consists of RGB values within [0, 255].
            return imagenet_utils.preprocess_input(x, mode='tf')


	@tf_export('keras.applications.xception.preprocess_input')
	def preprocess_input(x):
            x: a 4D numpy array consists of RGB values within [0, 255].
	    return imagenet_utils.preprocess_input(x, mode='tf')


        @tf_export('keras.applications.densenet.preprocess_input')
        def preprocess_input(x, data_format=None):
            x: a 3D or 4D numpy array consists of RGB values within [0, 255].
	    data_format: data format of the image tensor.
	    return imagenet_utils.preprocess_input(x, data_format, mode='torch')

        """



        # TODO: Check if x can also be a tensor or it should be a numpy array only

        # logging.debug('model_name {}'.format(self.config.model_name))
        if self.config.model_name == 'vgg16':
            image = preprocess_input_vgg16(x=image, data_format='channels_last')
            # image = tf.subtract(image, 116.779) # Zero-center by mean pixel
            # image = tf.reverse(image, axis=[2]) # 'RGB'->'BGR'

        elif self.config.model_name == 'xception':
            image = preprocess_input_xception(x=image)

        elif self.config.model_name == 'resnet50':
            image = preprocess_input_resnet50(x=image, data_format='channels_last')

        elif self.config.model_name == 'inception_resnet_v2':
            image = preprocess_input_inception_resnet_v2(x=image)

        elif self.config.model_name == 'densenet121':
            image = preprocess_input_densenet(x=image, data_format='channels_last')

        elif self.config.model_name == 'densenet169':
            image = preprocess_input_densenet(x=image, data_format='channels_last')

        elif self.config.model_name == 'densenet201':
            image = preprocess_input_densenet(x=image, data_format='channels_last')

        else:
            logging.error('Unknown model_name {}'.format(model_name))
            exit(1)

        return image


    def augment_data(self, image):
        # image = tf.slice(input_tensor, [i, 0, 0, 0], [1, 32, 32, 3])[0]
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=63)
        image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        image = tf.expand_dims(image, 0)
        return image


    def parse_fn(self, serialized):
        """
        1. Parse tf.Example
        2. jpeg decode
        3. image augmentations

        Parse TFRecords and perform simple data augmentation."
        https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/python/ops/parsing_ops.py
        `tf.float32` (`FloatList`), `tf.int64` (`Int64List`), and `tf.string` (`BytesList`) are supported.
        """
        # Define a dict with the data-names and types we expect to find in the TFRecords file.
        features = \
            {
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)
            }

        # Parse the serialized data so we get a dict with our data.
        parsed_example = tf.parse_single_example(serialized=serialized,
                                                 features=features)

        # Get the image as raw bytes.
        image_shape = tf.stack([self.config.tfr_image_height, self.config.tfr_image_width, self.config.tfr_image_channels])
        image_raw = parsed_example['image']

        # Decode the raw bytes so it becomes a tensor with type.
        image = tf.decode_raw(image_raw, tf.uint8)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, image_shape)

        # Preprocess the data
        image = self.preprocess_data(image)

        # Augmentation of the data
        # image = self.augment_data(image)

        # Get the label associated with the image.
        label = tf.cast(parsed_example['label'], tf.float32)

        return image, label


    # def input_fn(self, file_pattern, train, batch_size=32, buffer_size=2048):
    # def input_fn(self, file_pattern, mode, mode_ds='train_ds', batch_size=32, buffer_size=2048, num_repeat=None):
    def input_fn(self, file_pattern, shuffle=True, buffer_size=2048, num_repeat=None, batch_size=32, num_take=None):
        """
        Args:
            file_pattern:   Path with pattern of TFRecords. Eg: '/home/*.tfr'
            train:          Boolean whether training (True) or testing (False).
            batch_size:     Return batches of this size.
            buffer_size:    Read buffers of this size. The random shuffling
                            is done on the buffer, so it must be big enough.

        ETL: Extract, Transform, Load
        https://www.tensorflow.org/versions/master/performance/datasets_performance
        https://www.youtube.com/watch?v=SxOsJPaxHME
        """

        ## EXTRACT data
        filenames = tf.data.Dataset.list_files(file_pattern)

        if tf.__version__ >= "1.7.0":
            dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=self.config.data_gen_num_parallel_reads)
        else:
            dataset = tf.data.TFRecordDataset(filenames=filenames)


        ## TRANSFORM data to prepare for training
        if shuffle:
            dataset = dataset.shuffle(buffer_size=buffer_size)

        # if mode == 'train':
        #     # If training then read a buffer of the given size and randomly shuffle it.
        #     dataset = dataset.shuffle(buffer_size=buffer_size)

        #     # Allow infinite reading of the data.
        #     num_repeat_mode = None

        # elif mode == 'eval':
        #     if mode_ds =
        #     # If testing then don't shuffle the data. Only go through the data once.
        #     num_repeat_mode = 1

        # else:
        #     logging.error('Unknown mode {}'.format(mode))
        #     exit(1)


        # if num_repeat != None:
        #     num_repeat_mode = num_repeat
        # logging.debug('num_repeat_mode {}'.format(num_repeat_mode))

        # Repeat the dataset the given number of times.
        # dataset = dataset.repeat(num_repeat_mode)
        dataset = dataset.repeat(num_repeat)

        if num_take != None:
            dataset = dataset.take(num_take)

        # Parse the serialized data in the TFRecords files.
        # This returns TensorFlow tensors for the image and labels.
        # num_parallel_calls: recommend using the number of available CPU cores for its value.
        if (tf.__version__ >= '1.7.0'):
            dataset = dataset.map(self.parse_fn, num_parallel_calls=self.config.data_gen_num_parallel_calls)
        else:
            dataset = dataset.map(self.parse_fn)

        # Get a batch of data with the given size.
        dataset = dataset.batch(batch_size)
        print('dataset', dataset)

        # Maximum number of elements that will be buffered
        # prefetch(n) (where n is the number of elements / batches consumed by a training step)
        # Prefetch at end to ensure everything above is pipelined with the accelerator training
        # Larger prefetch buffers also smooth over latency variability
        dataset = dataset.prefetch(buffer_size=self.config.data_gen_prefetch_buffer_size)


        ## LOAD data into the accelerator
        # Create an iterator for the dataset and the above modifications.
        iterator = dataset.make_one_shot_iterator()

        # Get the next batch of images and labels.
        images_batch, labels_batch = iterator.get_next()
        # logging.debug('images_batch {}'.format(images_batch))
        # logging.debug('labels_batch {}'.format(labels_batch))


        # The convolutional layers expect 4-rank tensors but images_batch is a 2-rank tensor, so reshape it.
        # images_batch = tf.reshape(images_batch, [-1, self.config.tfr_image_height, self.config.tfr_image_width, self.config.tfr_image_channels])
        # logging.debug('images_batch {}'.format(images_batch))

        # result = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), images)


        # Preprocess image batch
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/_impl/keras/applications/imagenet_utils.py
        # images_batch = imagenet_utils.preprocess_input(images_batch, data_format=None, mode='torch')
        # images_batch = imagenet_utils.preprocess_input(images_batch, data_format=None, mode='tf')
        # logging.debug('images_batch {}'.format(images_batch))

        # Preprocess image
        # data_preprocessing_fn = lambda x: self.data_preprocessing(x)
        # images_batch = tf.map_fn(data_preprocessing_fn, images_batch)
        # # images_batch = tf.map_fn(fn=lambda x: self.data_preprocessing(x), images_batch)
        # print('images_batch', images_batch)

        # Augments image using flip, brightness, contrast, etc
        # image = self.data_augmentation(image)



        ## Features
        # The input-function must return a dict wrapping the images.
        x_key = self.config.model_name + '_input'
        feature_dict = {x_key: images_batch}

        ## Labels
        labels_batch = tf.cast(labels_batch, tf.int64)
        # label = labels_batch
        label = tf.one_hot(labels_batch, self.config.num_classes)


        # Print for debugging
        # if self.config.debug_tf_print:
        #     x['image'] = tf.Print(x['image'], [tf.shape(x['image'])], '\nTF x\n', summarize=20)
        #     y = tf.Print(y, [y, tf.shape(y)], '\nTF y\n', summarize=20)

        return feature_dict, label


    def read_dataset(self, dataset_path):
        image_paths_list = []
        gt_labels = []

        for label_name, label_no in self.config.labels.items():
            # Read image paths
            image_paths = utils_image.get_images_path_list_from_dir(
                                                os.path.join(dataset_path, label_name),
                                                img_format=self.config.dataset_path_image_format)
            images_count = len(image_paths)
            image_paths_list = image_paths_list + image_paths

            # Create labels
            gt_labels = gt_labels + [label_no]*images_count

        return image_paths_list, gt_labels


if __name__ == '__main__':
    try:
        args = utils.get_args()
        config = process_config(args)
    except:
        print("missing or invalid arguments")
        args={}
        args['config_file'] = 'configs/config_densenet.json'
        args['mode'] = 'train'
        args['mode_ds'] = 'train_ds'
        args['debug'] = 0
        args = Bunch(args)
        config = process_config(args)

    # Initialize Logger
    utils.logger_init(config, logging.DEBUG)


    filenames_regex = os.path.join(config.tfrecords_path_train, '*.tfr')
    filenames = glob.glob(filenames_regex)
    logging.debug('filenames {}'.format(filenames))

    data_generator_densenet = DataGeneratorDensenet(config)
    # next_batch = data_generator_densenet.input_fn(filenames=filenames, mode='train')
    next_batch = data_generator_densenet.input_fn(
                                file_pattern=os.path.join(config.tfrecords_path_train, '*.tfr'),
                                mode=config.mode,
                                batch_size=config.batch_size,
                                buffer_size=config.data_gen_buffer_size
                                )

    logging.debug('next_batch {}'.format(next_batch))

    with tf.Session() as sess:

        images_batch, labels_batch = sess.run(next_batch)

        image = images_batch['densenet121_input'][0]
        label = labels_batch[0]
        print('labels_batch', labels_batch)
        print(image.shape)
        print('label', label)
        print('label', type(label))

        img = k_image.array_to_img(image)

        img.show()




## Code for future use
# VGG16:
# image = tf.subtract(image, 116.779) # Zero-center by mean pixel
# image = tf.reverse(image, axis=[2]) # 'RGB'->'BGR'


