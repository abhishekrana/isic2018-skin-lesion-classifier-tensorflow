##########################################################################################
# Class for creating Tensorflow Records
#  "labels":{"airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "deer": 4, "dog": 5, "frog": 6, "horse": 7, "ship": 8, "truck": 9},
##########################################################################################
import os
os.sys.path.append('./')
os.sys.path.append('../')
os.sys.path.append('Classification')

print(os.getcwd())


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

        ## Read dataset
        image_paths_train, gt_labels_train = self.read_dataset(self.config.dataset_path_train)
        image_paths_test, gt_labels_test = self.read_dataset(self.config.dataset_path_test)

        image_paths_train, gt_labels_train  = utils.shuffle_data(image_paths_train, gt_labels_train)
        image_paths_test, gt_labels_test  = utils.shuffle_data(image_paths_test, gt_labels_test)

        ## For debugging on smaller dataset
        if config.debug_train_images_count != 0:
            image_paths_train = image_paths_train[0:config.debug_train_images_count]
            gt_labels_train = gt_labels_train[0:config.debug_train_images_count]
        if config.debug_test_images_count != 0:
            image_paths_test = image_paths_test[0:config.debug_test_images_count]
            gt_labels_test = gt_labels_test[0:config.debug_test_images_count]


        ## Convert train dataset to TFRecord
        data = {
            'image': image_paths_train,
            'label': gt_labels_train
            }
        self.dataset_to_tfrecords(data=data, output_path=self.config.tfrecords_path_train)


        ## Convert test dataset to TFRecord
        data = {
            'image': image_paths_test,
            'label': gt_labels_test
            }
        self.dataset_to_tfrecords(data=data, output_path=self.config.tfrecords_path_test)


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
            print('ERROR: Unknown type of data: {}'.format(type(value[0])))
            exit(1)



### MAIN ###
if __name__ == '__main__':
    # args = utils.get_args()
    # config = process_config(args.config)

    config_file = 'configs/config_densenet.json'
    config = process_config(config_file)

    tfrecords_cfiar10 = TFRecordsDensenet(config)


