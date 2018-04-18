##########################################################################################
# Class for creating Tensorflow Records
# https://challenge2018.isic-archive.com/task1/
##########################################################################################

import os
os.sys.path.append('./')

from base.base_tfrecords import BaseTFRecords
import tensorflow as tf
import numpy as np
from PIL import Image
import pudb

import utils.utils as utils
import utils.utils_image as utils_image
from utils.config import process_config


class TFRecordsUnet(BaseTFRecords):

    def __init__(self, config):
        super(TFRecordsUnet, self).__init__(config)


        ## Read dataset
        image_paths_train, gt_image_paths_train = self.read_dataset(self.config.dataset_path_train_processed,
                                                                    self.config.dataset_path_train_processed_gt)
        image_paths_test, gt_image_paths_test = self.read_dataset(self.config.dataset_path_test_processed,
                                                                  self.config.dataset_path_test_processed_gt)


        ## For debugging on smaller dataset
        image_paths_train = image_paths_train[0:10]
        gt_image_paths_train = gt_image_paths_train[0:10]
        image_paths_test = image_paths_test[0:10]
        gt_image_paths_test = gt_image_paths_test[0:10]


        ## Convert train dataset to TFRecord
        data = {
            'image': image_paths_train,
            'gt_image': gt_image_paths_train
            }
        self.dataset_to_tfrecords(data=data, output_path=self.config.tfrecords_path_train)


        ## Convert test dataset to TFRecord
        data = {
            'image': image_paths_test,
            'gt_image': gt_image_paths_test
            }
        self.dataset_to_tfrecords(data=data, output_path=self.config.tfrecords_path_test)


    def read_dataset(self, dataset_path, gt_dataset_path):
        gt_image_paths_list = []

        # Read image paths
        image_paths_list = utils_image.get_images_path_list_from_dir(dataset_path, img_format='jpg')

        # Generate ground-truth image paths
        for image_path_name in image_paths_list:
            gt_image_name = os.path.basename(image_path_name).split('processed')[0] + 'segmentation_processed.png'
            gt_image_path_name = os.path.join(gt_dataset_path, gt_image_name)
            gt_image_paths_list.append(gt_image_path_name)

        return image_paths_list, gt_image_paths_list


    def wrap_data(self, key, value, output_path):
        if key == 'image':
            # Pass output_path as param in order to save images
            return self.wrap_data_image(value, output_path)

        elif key == 'gt_image':
            # Pass output_path as param in order to save images
            return self.wrap_data_image(value, output_path)

        elif key == 'label':
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        else:
            print('ERROR: Unknown type of data: {}'.format(type(value[0])))
            exit(1)



### MAIN ###
if __name__ == '__main__':
    args = utils.get_args()
    config = process_config(args.config)

    tfrecords_unet = TFRecordsUnet(config)


