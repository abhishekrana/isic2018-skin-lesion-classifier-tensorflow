##########################################################################################
# Base class for creating Tensorflow Records
##########################################################################################
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import pudb

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


    def wrap_data(self, key, value, output_path):
        NotImplementedError


    def wrap_data_image(self, image_path, output_path=None):

        ## Convert the image to raw bytes.
        img_size = (self.config.tfr_image_width, self.config.tfr_image_height)

        img = Image.open(image_path)
        img = img.resize(size=img_size, resample=Image.LANCZOS)
        img = np.array(img)

        if output_path is not None:
            img_path_name = os.path.join(os.path.dirname(output_path), os.path.basename(image_path))
            utils_image.save_image(img, img_path_name)

        img_bytes = img.tostring()

        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes]))


    def dataset_to_tfrecords(self, data, output_path):

        # Number of images. Used when printing the progress.
        num_images = len(data['image'])

        # TODO: aSk: Handle case when images do not fit batch
        batch_size = self.config.tfr_images_per_record
        iters = int(num_images/batch_size)

        for iter_no in range(iters):
            idx_start = iter_no * batch_size
            idx_end = idx_start + batch_size

            output_path_mod = os.path.join(output_path, 'record_' + str(iter_no) + '.tfr')
            self.create_tfrecord(data, idx_start, idx_end, output_path_mod, iter_no)

            # Print the percentage-progress.
            utils.print_progress(count=idx_start, total=num_images)


    def create_tfrecord(self, data, idx_start, idx_end, output_path, iter_no):

        # Open a TFRecordWriter for the output-file.
        with tf.python_io.TFRecordWriter(output_path) as writer:

            for i in range(idx_start, idx_end):
                tf_data = {}

                # Create a dict with the data we want to save in the TFRecords file.
                for key, value in data.items():
                    tf_data[key] = self.wrap_data(key, value[i], output_path)

                # Wrap the data as TensorFlow Features.
                feature = tf.train.Features(feature=tf_data)

                # Wrap again as a TensorFlow Example.
                example = tf.train.Example(features=feature)

                # Serialize the data.
                serialized = example.SerializeToString()

                # Write the serialized data to the TFRecords file.
                writer.write(serialized)


