##########################################################################################
# Class for generating batches of data using Tensorflow Records
##########################################################################################

import os
os.sys.path.append('./')

from base.base_data import BaseData
import numpy as np
import tensorflow as tf
import glob
import pudb

import utils.utils as utils
import utils.utils_image as utils_image
from utils.config import process_config

class DataGeneratorUnet(BaseData):
    def __init__(self, config):
        super(DataGeneratorUnet, self).__init__(config)


    def parse_fn(self, serialized):
        """Parse TFRecords and perform simple data augmentation."
        """
        # Define a dict with the data-names and types we expect to find in the TFRecords file.
        # It is a bit awkward that this needs to be specified again, because it could have been
        # written in the header of the TFRecords file instead.
        features = \
            {
                'image': tf.FixedLenFeature([], tf.string),
                'gt_image': tf.FixedLenFeature([], tf.string)
            }

        # Parse the serialized data so we get a dict with our data.
        parsed_example = tf.parse_single_example(serialized=serialized,
                                                 features=features)

        # Get the image as raw bytes.
        image_raw = parsed_example['image']

        # Decode the raw bytes so it becomes a tensor with type.
        image = tf.decode_raw(image_raw, tf.uint8)

        # The type is now uint8 but we need it to be float.
        image = tf.cast(image, tf.float32)

        # Augments image using slice, reshape, resize_bilinear
        # image = _augment_helper(image)


        # Get the gt_image as raw bytes.
        gt_image_raw = parsed_example['gt_image']

        # Decode the raw bytes so it becomes a tensor with type.
        gt_image = tf.decode_raw(gt_image_raw, tf.uint8)

        # The type is now uint8 but we need it to be float.
        gt_image = tf.cast(gt_image, tf.float32)

        # Augments gt_image using slice, reshape, resize_bilinear
        # gt_image = _augment_helper(gt_image)


        # The image and gt_image/label are now correct TensorFlow types.
        return image, gt_image


    def input_fn(self, filenames, train, batch_size=32, buffer_size=2048):
        # Args:
        # filenames:   Filenames for the TFRecords files.
        # train:       Boolean whether training (True) or testing (False).
        # batch_size:  Return batches of this size.
        # buffer_size: Read buffers of this size. The random shuffling
        #              is done on the buffer, so it must be big enough.

        # Create a TensorFlow Dataset-object which has functionality
        # for reading and shuffling data from TFRecords files.
        dataset = tf.data.TFRecordDataset(filenames=filenames)

        # Parse the serialized data in the TFRecords files.
        # This returns TensorFlow tensors for the image and labels.
        # num_parallel_calls: recommend using the number of available CPU cores for its value.
        dataset = dataset.map(self.parse_fn, num_parallel_calls=self.config.num_parallel_calls)

        if train:
            # If training then read a buffer of the given size and randomly shuffle it.
            dataset = dataset.shuffle(buffer_size=buffer_size)

            # Allow infinite reading of the data.
            num_repeat = None
        else:
            # If testing then don't shuffle the data.
            # Only go through the data once.
            num_repeat = 1

        # Repeat the dataset the given number of times.
        dataset = dataset.repeat(num_repeat)

        # Get a batch of data with the given size.
        dataset = dataset.batch(batch_size)

        # Maximum number of elements that will be buffered
        # prefetch(n) (where n is the number of elements / batches consumed by a training step)
        dataset = dataset.prefetch(buffer_size=self.config.prefetch_buffer_size)

        # Create an iterator for the dataset and the above modifications.
        iterator = dataset.make_one_shot_iterator()

        # Get the next batch of images and gt_images/labels.
        images_batch, gt_images_batch = iterator.get_next()


        # The input-function must return a dict wrapping the images.
        x = {'image': images_batch}
        y = gt_images_batch


        # Print images_batch shape for debugging
        # images_batch = tf.Print(images_batch, [tf.shape(images_batch)], '\nTF images_batch shape\n', summarize=10)

        return x, y


    # TODO: Move/Refactor this redundant code to some other file
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


if __name__ == '__main__':
    args = utils.get_args()
    config = process_config(args.config)

    filenames_regex = os.path.join(config.tfrecords_path_train, '*.tfr')
    filenames = glob.glob(filenames_regex)

    data_generator_unet = DataGeneratorUnet(config)
    data_generator_unet.input_fn(filenames=filenames, train=True)


