##########################################################################################
# Class for creating Tensorflow Records
# labels = {'forkey':0, 'knifey':1, 'spoony':2}
##########################################################################################

from base.base_tfrecords import BaseTFRecords
import tensorflow as tf
import os
import numpy as np
from PIL import Image
import pudb

import utils.utils as utils
import utils.utils_image as utils_image
from utils.config import process_config


class TFRecordsKnifeySpoony(BaseTFRecords):

    def __init__(self, config):
        super(TFRecordsKnifeySpoony, self).__init__(config)


        ## Read dataset
        image_paths_train, gt_labels_train = self.read_dataset(self.config.dataset_path_train)
        image_paths_test, gt_labels_test = self.read_dataset(self.config.dataset_path_test)


        ## For debugging on smaller dataset
        # image_paths_train = image_paths_train[0:10]
        # image_paths_test = image_paths_test[0:10]


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
                                                img_format='jpg')
            images_count = len(image_paths)
            image_paths_list = image_paths_list + image_paths

            # Create labels
            gt_labels = gt_labels + [label_no]*images_count

        return image_paths_list, gt_labels


    def wrap_data(self, key, value, output_path):
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
    args = utils.get_args()
    config = process_config(args.config)
    tfrecords_ks = TFRecordsKnifeySpoony(config)


