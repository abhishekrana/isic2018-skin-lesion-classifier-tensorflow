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



class DataGeneratorDensenet(BaseData):
    def __init__(self, config):
        super(DataGeneratorDensenet, self).__init__(config)


    def parse_fn(self, serialized):
        """Parse TFRecords and perform simple data augmentation."
        https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/python/ops/parsing_ops.py
        `tf.float32` (`FloatList`), `tf.int64` (`Int64List`), and `tf.string` (`BytesList`) are supported.
        """
        # Define a dict with the data-names and types we expect to find in the TFRecords file.
        # It is a bit awkward that this needs to be specified again, because it could have been
        # written in the header of the TFRecords file instead.
        features = \
            {
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)
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

        # Get the label associated with the image.
        label = parsed_example['label']

        # The image and label are now correct TensorFlow types.
        return image, label


    def input_fn(self, filenames, train, batch_size=32, buffer_size=2048):
        # Args:
        # filenames:   Filenames for the TFRecords files.
        # train:       Boolean whether training (True) or testing (False).
        # batch_size:  Return batches of this size.
        # buffer_size: Read buffers of this size. The random shuffling
        #              is done on the buffer, so it must be big enough.
        # https://www.tensorflow.org/versions/master/performance/datasets_performance
        # https://www.youtube.com/watch?v=SxOsJPaxHME

        ## EXTRACT data from storage
        # Create a TensorFlow Dataset-object which has functionality
        # for reading and shuffling data from TFRecords files.
        # files = tf.data.Dataset.list_files(file_pattern)
        if tf.__version__ == "1.7.0":
            dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=self.config.data_gen_num_parallel_reads)
        else:
            dataset = tf.data.TFRecordDataset(filenames=filenames)


        ## TRANSFORM data to prepare for training
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

        # Parse the serialized data in the TFRecords files.
        # This returns TensorFlow tensors for the image and labels.
        # num_parallel_calls: recommend using the number of available CPU cores for its value.
        if (tf.__version__ == '1.7.0'):
            dataset = dataset.map(self.parse_fn, num_parallel_calls=self.config.data_gen_num_parallel_calls)
        else:
            dataset = dataset.map(self.parse_fn)

        # Get a batch of data with the given size.
        dataset = dataset.batch(batch_size)

        # Maximum number of elements that will be buffered
        # prefetch(n) (where n is the number of elements / batches consumed by a training step)
        # Prefetch at end to ensure everything above is pipelined with the accelerator training
        # Larger prefetch buffers also smooth over latency variability
        dataset = dataset.prefetch(buffer_size=self.config.data_gen_prefetch_buffer_size)


        ## LOAD data into the accelerator
        # Create an iterator for the dataset and the above modifications.
        iterator = dataset.make_one_shot_iterator()

        # Get the next batch of images and labels.
        images_batch, labels_batch = iterator.get_next()
        print('labels_batch', labels_batch)


        # The convolutional layers expect 4-rank tensors but images_batch is a 2-rank tensor, so reshape it.
        images_batch = tf.reshape(images_batch, [-1, self.config.tfr_image_height, self.config.tfr_image_width, self.config.tfr_image_channels])

        # Convert labels to categorical format
        labels_batch_categorical = tf.one_hot(labels_batch, self.config.num_classes)

        # The input-function must return a dict wrapping the images.
        # x = {'input_1': images_batch}
        x = {'images_input': images_batch}
        y = labels_batch_categorical

        # Print for debugging
        # if self.config.debug_tf_print:
        #     x['image'] = tf.Print(x['image'], [tf.shape(x['image'])], '\nTF x\n', summarize=20)
        #     y = tf.Print(y, [y, tf.shape(y)], '\nTF y\n', summarize=20)

        return x, y


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

    # args = utils.get_args()
    # config = process_config(args.config)

    config_file = 'configs/config_densenet.json'
    # config_file = 'configs/config_cifar10.json'
    config = process_config(config_file)

    filenames_regex = os.path.join(config.tfrecords_path_train, '*.tfr')
    filenames = glob.glob(filenames_regex)
    print('filenames', filenames)

    data_generator_densenet = DataGeneratorDensenet(config)
    next_batch = data_generator_densenet.input_fn(filenames=filenames, train=True)
    print('next_batch', next_batch)

    with tf.Session() as sess:

        images_batch, labels_batch = sess.run(next_batch)

        print('images_batch:{} shape:{}'.format(images_batch, images_batch['images_input'].shape))
        print('labels_batch:{} shape:{}'.format(labels_batch, labels_batch.shape))

        image = tf.keras.preprocessing.image.array_to_img(images_batch['images_input'][0])
        image.show()

