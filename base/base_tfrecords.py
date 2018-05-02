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


    def dataset_to_tfrecords(self, image_paths, labels, output_path):

        ## Number of images. Used when printing the progress.
        num_images = len(image_paths)

        batch_size = self.config.tfr_images_per_record
        iters = int(num_images/batch_size)
        print('\nnum_images: {}'.format(num_images))
        print('batch_size: {}'.format(batch_size))

        idx_start = 0
        idx_end = 0
        for iter_no in range(iters):
            idx_start = iter_no * batch_size
            idx_end = idx_start + batch_size
            print('\nidx:[{}-{}]'.format(idx_start, idx_end))

            output_path_mod = os.path.join(output_path, 'record_' + str(iter_no) + '.tfr')
            self.create_tfrecord(image_paths, labels, idx_start, idx_end, output_path_mod, iter_no)

            # Print the percentage-progress.
            utils.print_progress(count=idx_start, total=num_images)

        ## For images < batch_size and 
        ## For images which do not fit the last batch
        idx_start = iters * batch_size
        idx_end = idx_start + (num_images % batch_size)
        print('\nidx:[{}-{}]'.format(idx_start, idx_end))
        if(num_images % batch_size):
            output_path_mod = os.path.join(output_path, 'record_' + str(iters) + '.tfr')
            self.create_tfrecord(image_paths, labels, idx_start, idx_end, output_path_mod)

            # Print the percentage-progress.
            # utils.print_progress(count=idx_start, total=num_images)

        print('\n')



    def wrap_bytes(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def wrap_int64(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def create_tfrecord(self, image_paths, labels, idx_start, idx_end, output_path):

        # Open a TFRecordWriter for the output-file.
        with tf.python_io.TFRecordWriter(output_path) as writer:

            for i in range(idx_start, idx_end):

                utils.print_progress(count=i, total=(idx_end-idx_start))

                image_path = image_paths[i]
                label = labels[i]

                # Load the image-file using matplotlib's imread function.
                img = Image.open(image_path)

                # Center crop and resize image
                # img = ImageOps.fit(img, (self.config.tfr_image_width, self.config.tfr_image_height), Image.LANCZOS, 0, (0.5, 0.5))
                # size: The requested size in pixels, as a 2-tuple: (width, height)
                img = img.resize(size=(self.config.tfr_image_width, self.config.tfr_image_height))

                img = np.array(img)

                if output_path is not None:
                    img_path_name = os.path.join(os.path.dirname(output_path), os.path.basename(image_path))
                    utils_image.save_image(img, img_path_name)

                # Convert the image to raw bytes.
                img_bytes = img.tostring()

                data = {
                    'image': self.wrap_bytes(img_bytes),
                    'label': self.wrap_int64(label)
                    }

                # Wrap the data as TensorFlow Features.
                feature = tf.train.Features(feature=data)

                # Wrap again as a TensorFlow Example.
                example = tf.train.Example(features=feature)

                # Serialize the data.
                serialized = example.SerializeToString()

                # Write the serialized data to the TFRecords file.
                writer.write(serialized)

    def rotate_image(image):
        image_path = Image.open(image)
        image_array = np.array(image_path)
        rotated_image1 = np.rot90(image_array)
        rotated_image2 = np.rot90(rotated_image1)
        rotated_image3 = np.rot90(rotated_image2)
        return [image_array, rotated_image1, rotated_image2, rotated_image3]
