
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys
import os
import random
import pudb

import utils.utils as utils
import utils.utils_image as utils_image



class TFRecordsGenerator():

    def __init__(self, config):

        utils.remove_dirs([config.tfrecords_path_train, config.tfrecords_path_test])
        utils.create_dirs([config.tfrecords_path_train, config.tfrecords_path_test])

        tfrecords_path_train = os.path.join(config.tfrecords_path_train, "train.tfrecords")
        tfrecords_path_test = os.path.join(config.tfrecords_path_test, "test.tfrecords")

        ## Train dataset should have 2600 images but currently has 2591 images due to some corrupted files
        image_paths_train = self.read_dataset(config.dataset_path_train)
        # image_paths_test = self.read_dataset(config.dataset_path_test)


        ## Split train dataset into train and test. Train: 2000; Test: 590
        ## Do not change order
        random.shuffle(image_paths_train)
        image_paths_test = image_paths_train[2001:]
        image_paths_train = image_paths_train[0:2000]

        gt_image_paths_train = []
        for image_path_train in image_paths_train:
            gt_image_paths_train.append(os.path.join(config.dataset_path_train_gt,
                                         os.path.basename(image_path_train).split('.jpg')[0] + '_segmentation.png'))

        gt_image_paths_test = []
        for image_path_test in image_paths_test:
            gt_image_paths_test.append(os.path.join(config.dataset_path_train_gt,
                                         os.path.basename(image_path_test).split('.jpg')[0] + '_segmentation.png'))


        # self.dataset_to_tfrecords(image_paths=image_paths_train, labels=cls_train, out_path=tfrecords_path_train)

        # TODO: _aSk Just for debugging
        image_paths_train = image_paths_train[0:100]
        gt_image_paths_train = gt_image_paths_train[0:100]
        self.dataset_to_tfrecords(config=config, image_paths=image_paths_train, gt_image_paths=gt_image_paths_train, out_path=tfrecords_path_train)


        # TODO: _aSk Just for debugging
        image_paths_test = image_paths_test[0:50]
        gt_image_paths_test = gt_image_paths_test[0:50]
        self.dataset_to_tfrecords(config=config, image_paths=image_paths_test, gt_image_paths=gt_image_paths_test, out_path=tfrecords_path_test)


    def read_dataset(self, dataset_path):
        images_paths_list = utils_image.get_images_path_list_from_dir(dataset_path)
        return images_paths_list


    def wrap_int64(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


    def wrap_bytes(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


    def dataset_to_tfrecords(self, config, image_paths, gt_image_paths, out_path):
        """
        Args:
        image_paths       List of file-paths for the images.
        gt_image_paths    List of file-paths for the ground truth images.
        out_path          File-path for the TFRecords output file.
        """

        image_size = (config.tfr_image_width, config.tfr_image_height)
        print("\nConverting dataset to TFRecords: {}. Resizing images to {}".format(out_path, image_size))


        # Number of images. Used when printing the progress.
        num_images = len(image_paths)


        # TODO: aSk: Handle if images do not fit batch
        batch_size = config.tfr_images_count
        iters = int(num_images/batch_size)

        for iter_no in range(iters):
            idx_start = iter_no * batch_size
            idx_end = idx_start + batch_size

            out_path_mod = out_path.split('.tfrecords')[0] + '_' + str(iter_no) + '.tfrecords'
            self.create_tfrecords(image_paths[idx_start:idx_end], gt_image_paths[idx_start:idx_end], out_path_mod, iter_no, image_size)

            # Print the percentage-progress.
            utils.print_progress(count=idx_start, total=num_images)


    def create_tfrecords(self, image_paths, gt_image_paths, out_path, iter_no, image_size):

        # Open a TFRecordWriter for the output-file.
        with tf.python_io.TFRecordWriter(out_path) as writer:

            for i, (path, gt_path) in enumerate(zip(image_paths, gt_image_paths)):

                # Load the image-file using matplotlib's imread function.
                # Load image
                img = utils_image.load_image(path, image_size)
                gt_img = utils_image.load_image(gt_path, image_size)
                # img = imread(path)
                # gt_img = imread(gt_path)

                # Convert the image to raw bytes.
                img_bytes = img.tostring()
                gt_img_bytes = gt_img.tostring()

                # Create a dict with the data we want to save in the
                # TFRecords file. You can add more relevant data here.
                data = \
                    {
                        'image': self.wrap_bytes(img_bytes),
                        'gt_image': self.wrap_bytes(gt_img_bytes),
                        # 'label': self.wrap_int64(label)
                    }

                # Wrap the data as TensorFlow Features.
                feature = tf.train.Features(feature=data)

                # Wrap again as a TensorFlow Example.
                example = tf.train.Example(features=feature)

                # Serialize the data.
                serialized = example.SerializeToString()

                # Write the serialized data to the TFRecords file.
                writer.write(serialized)





