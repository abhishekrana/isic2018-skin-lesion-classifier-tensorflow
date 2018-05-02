##########################################################################################
# Class for creating Tensorflow Records
#  "labels":{"airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "deer": 4, "dog": 5, "frog": 6, "horse": 7, "ship": 8, "truck": 9},
##########################################################################################
import os
os.sys.path.append('./')
os.sys.path.append('../')


from base.base_tfrecords import BaseTFRecords
import tensorflow as tf
import numpy as np
from PIL import Image
import pudb

import utils.utils as utils
import utils.utils_image as utils_image
from utils.config import process_config


class TFRecordsDensenet(BaseTFRecords):

    def __init__(self, config):
        super(TFRecordsDensenet, self).__init__(config)

        utils.create_dirs([config.tfrecords_path_train, config.tfrecords_path_val, config.tfrecords_path_test])

        ## Read dataset
        image_paths_train, labels_train = self.read_dataset(self.config.dataset_path_train)
        image_paths_val, labels_val = self.read_dataset(self.config.dataset_path_val)
        image_paths_test, labels_test = self.read_dataset(self.config.dataset_path_test)

        image_paths_train, labels_train  = utils.shuffle_data(image_paths_train, labels_train)
        image_paths_val, labels_val  = utils.shuffle_data(image_paths_val, labels_val)
        image_paths_test, labels_test  = utils.shuffle_data(image_paths_test, labels_test)

        ## For debugging on smaller dataset
        if config.debug_train_images_count != 0:
            image_paths_train = image_paths_train[0:config.debug_train_images_count]
            labels_train = labels_train[0:config.debug_train_images_count]
        if config.debug_val_images_count != 0:
            image_paths_val = image_paths_val[0:config.debug_val_images_count]
            labels_val = labels_val[0:config.debug_val_images_count]
        if config.debug_test_images_count != 0:
            image_paths_test = image_paths_test[0:config.debug_test_images_count]
            labels_test = labels_test[0:config.debug_test_images_count]


        ## Convert train dataset to TFRecord
        self.dataset_to_tfrecords(image_paths_train, labels_train, output_path=self.config.tfrecords_path_train)

        ## Convert val dataset to TFRecord
        self.dataset_to_tfrecords(image_paths_val, labels_val, output_path=self.config.tfrecords_path_val)

        ## Convert test dataset to TFRecord
        self.dataset_to_tfrecords(image_paths_test, labels_test, output_path=self.config.tfrecords_path_test)


    def read_dataset(self, dataset_path):
        image_paths_list = []
        labels = []

        for label_name, label_no in self.config.labels.items():
            # Read image paths
            image_paths = utils_image.get_images_path_list_from_dir(
                                                os.path.join(dataset_path, label_name),
                                                img_format=self.config.dataset_path_image_format)
            images_count = len(image_paths)
            image_paths_list = image_paths_list + image_paths

            # Create labels
            labels = labels + [label_no]*images_count

        return image_paths_list, labels


    def wrap_data(self, key, value, output_path):
        """
        https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/python/ops/parsing_ops.py
        `tf.float32` (`FloatList`), `tf.int64` (`Int64List`), and `tf.string` (`BytesList`) are supported.
        """
        if key == 'image':
            # Pass output_path as param in order to save images
            return self.wrap_data_image(value, output_path)

        elif key == 'label':
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        else:
            log.error('ERROR: Unknown type of data: {}'.format(type(value[0])))
            exit(1)



### MAIN ###
if __name__ == '__main__':

    try:
        args = utils.get_args()
        config = process_config(args)
    except:
        print("missing or invalid arguments")
        config_file = 'configs/config_densenet.json'
        config = process_config(args)

    # Initialize Logger
    utils.logger_init(config, logging.DEBUG) 

    tfrecords_cfiar10 = TFRecordsDensenet(config)
