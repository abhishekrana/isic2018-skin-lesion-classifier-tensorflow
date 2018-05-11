##########################################################################################
# Class for creating Tensorflow Records
##########################################################################################
import os
os.sys.path.append('./')
os.sys.path.append('../')

from base.base_tfrecords import BaseTFRecords
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import pudb
import sklearn
import pickle

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


        ## Shuffle data
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


    def dataset_to_tfrecords(self, image_paths, labels, output_path):

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

        # For images < batch_size and 
        # For images which do not fit the last batch
        idx_start = iters * batch_size
        idx_end = idx_start + (num_images % batch_size)
        print('\nidx:[{}-{}]'.format(idx_start, idx_end))
        if(num_images % batch_size):
            output_path_mod = os.path.join(output_path, 'record_' + str(iters) + '.tfr')
            self.create_tfrecord(image_paths, labels, idx_start, idx_end, output_path_mod)

            # Print the percentage-progress.
            # utils.print_progress(count=idx_start, total=num_images)

        print('\n')

    def create_tfrecord(self, image_paths, labels, idx_start, idx_end, output_path):

        # Open a TFRecordWriter for the output-file.
        with tf.python_io.TFRecordWriter(output_path) as writer:

            for i in range(idx_start, idx_end):

                utils.print_progress(count=i, total=(idx_end-idx_start))

                image_path = image_paths[i]
                label = labels[i]

                # Load images
                img = Image.open(image_path)

                # TODO:
                # Center crop and resize image. size: The requested size in pixels, as a 2-tuple: (width, height)
                img = ImageOps.fit(img, (self.config.tfr_image_width, self.config.tfr_image_height), Image.LANCZOS, 0, (0.5, 0.5))
                # img = img.resize(size=(self.config.tfr_image_width, self.config.tfr_image_height))

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


### MAIN ###
if __name__ == '__main__':

    try:
        args = utils.get_args()
        config = process_config(args)
    except:
        print("missing or invalid arguments")
        args={}
        args['config_file'] = 'configs/config_densenet.json'
        args['mode'] = 'ds'
        args = Bunch(args)
        config = process_config(args)


    # Initialize Logger
    utils.logger_init(config, logging.DEBUG) 

    tfrecords_densenet = TFRecordsDensenet(config)

