##########################################################################################
# Base class for creating Tensorflow Records
##########################################################################################
import tensorflow as tf
import numpy as np
import os
from PIL import Image, ImageOps
import logging
import pudb
import cv2

import utils.utils as utils
import utils.utils_image as utils_image

class BaseTFRecords():

    def __init__(self, config):

        self.config = config

        ## Setup
        utils.remove_dirs([config.tfrecords_path_train, config.tfrecords_path_test])
        utils.create_dirs([config.tfrecords_path_train, config.tfrecords_path_test])


    def read_dataset(self, dataset_path):
        raise NotImplementedError


    def wrap_bytes(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


    def wrap_int64(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


    def rotate_image(image):
        image_path = Image.open(image)
        image_array = np.array(image_path)
        rotated_image1 = np.rot90(image_array)
        rotated_image2 = np.rot90(rotated_image1)
        rotated_image3 = np.rot90(rotated_image2)
        return [image_array, rotated_image1, rotated_image2, rotated_image3]
