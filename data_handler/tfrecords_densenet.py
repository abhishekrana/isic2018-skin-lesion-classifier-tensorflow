##########################################################################################
# Class for creating Tensorflow Records
##########################################################################################
import os
os.sys.path.append('./')
os.sys.path.append('../')

from base.base_tfrecords import BaseTFRecords
import cv2
import numpy as np
from PIL import Image, ImageOps
import pudb
import sklearn
import pickle
import shutil
import Augmentor
from imgaug import augmenters as iaa

import tensorflow as tf
from tensorflow.python.keras.preprocessing import image as k_image

import utils.utils as utils
import utils.utils_image as utils_image
from utils.config import process_config


class TFRecordsDensenet(BaseTFRecords):

    def __init__(self, config):
        super(TFRecordsDensenet, self).__init__(config)

        utils.remove_dirs([config.dataset_path_train_aug])
        utils.create_dirs([config.tfrecords_path_train, config.tfrecords_path_val, config.tfrecords_path_test,
            config.dataset_path_train_aug])

        ## Generate augmented dataset
        # self.data_augmentation_v1(self.config.dataset_path_train)
        # self.data_augmentation_v2()


        ## Read dataset
        image_paths_orig_train, labels_orig_train = self.read_dataset(self.config.dataset_path_train)
        image_paths_train_aug, labels_train_aug = self.read_dataset(self.config.dataset_path_train_aug)
        image_paths_train = image_paths_orig_train + image_paths_train_aug
        labels_train = labels_orig_train + labels_train_aug

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


    def data_augmentation_v1(self, dataset_path):
        print('Generating augmented images...\n')

        image_paths_list = []
        labels = []

        augmentations_dict = {
            'horizontal-flip'           : 1,
            'random-crop'               : 1,
            'gaussian-blur'             : 1,
            'contrast-norm'             : 1,
            'additive-gaussian-noise'   : 1,
        }
        augmentations_type = [ k for k,v in augmentations_dict.items() if v == 1 ]

        for augmentation_type in augmentations_type:
            print('augmentation_type', augmentation_type)

            for label_name, label_no in self.config.labels.items():

                # Read image paths
                image_paths = utils_image.get_images_path_list_from_dir(
                                                    os.path.join(dataset_path, label_name),
                                                    img_format=self.config.dataset_path_image_format)
                image_paths = image_paths[0:5]

                for idx, image_path in enumerate(image_paths):

                    img = Image.open(image_paths[idx])
                    img = np.array(img)

                    # TODO: Do center cropping
                    # img = cv2.imread(image_paths[idx])
                    # img = cv2.resize(img, (224, 224))

                    img_aug = self.data_augmentation_sequence(img, augmentation_type)

                    os.makedirs(os.path.join(self.config.dataset_path_train_aug, label_name), exist_ok=True)
                    img_name = os.path.basename(image_path).rsplit('.', 1)[0]
                    img_ext = os.path.basename(image_path).rsplit('.', 1)[1]

                    img_path_name = os.path.join(self.config.dataset_path_train_aug, label_name, img_name + '_' + augmentation_type + '.' + img_ext)

                    utils_image.save_image(img_aug, img_path_name)
                    print('img_path_name', img_path_name, idx)

                    ## Save original image for manual comparison
                    # img_path_name_orig = os.path.join(self.config.dataset_path_train_aug, label_name, img_name + '.' + img_ext)
                    # utils_image.save_image(img, img_path_name_orig)



    def data_augmentation_sequence(self, image, augmentation_type):
                
        ## Horizontal flips
        if augmentation_type == 'horizontal-flip':
            seq = iaa.Sequential([
                iaa.Fliplr(1)
            ])

        ## Random Crops
        if augmentation_type == 'random-crop':
            seq = iaa.Sequential([
                iaa.Crop(percent=(0, 0.25)),
            ])

        ## Small gaussian blur with random sigma between 0 and 0.5. But we only blur about 50% of all images.
        if augmentation_type == 'gaussian-blur':
            seq = iaa.Sequential([
            iaa.Sometimes(1.0, iaa.GaussianBlur(sigma=(0, 0.5))),
            ])

        ## Strengthen or weaken the contrast in each image.
        if augmentation_type == 'contrast-norm':
            seq = iaa.Sequential([
            iaa.ContrastNormalization((0.75, 1.0)),
            ])

        ## Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND channel. 
        # This can change the color (not only brightness) of the pixels.
        if augmentation_type == 'additive-gaussian-noise':
            seq = iaa.Sequential([
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            ])

            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel, which can end up changing the color of the images.
            # iaa.Multiply((0.8, 1.2), per_channel=0.2),

            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.

        # ], random_order=True)  # apply augmenters in random order
        # ])  # apply augmenters in random order


        # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
        # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
        # sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        # seq = iaa.Sequential(
        #     [
        #         iaa.SomeOf((0, 5),
        #         [
        #             sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
        #         ],
        #         random_order=True
        #         )
        #     ],
        #     random_order=True)


        image = cv2.convertScaleAbs(image, alpha=(255.0/255.0))
        image_aug = seq.augment_image(image)

        # img = k_image.array_to_img(image_aug)
        # img.show()

        return image_aug


    def data_augmentation_v2(self):

        ## Create pipeline
        p = Augmentor.Pipeline(source_directory=self.config.dataset_path_train, 
                output_directory=os.path.join('../../../', self.config.dataset_path_train_aug))

        # p.rotate(probability=0.1, max_left_rotation=0, max_right_rotation=0)

        ## Add Operations to the Pipeline
        # p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
        # p.zoom(probability=0.3, min_factor=1.1, max_factor=1.6)

        # p.crop_by_size(probability=1.0, width=self.config.tfr_image_width, height=self.config.tfr_image_height, centre=True)

        # Good values for parameters are between 2 and 10 for the grid width and height, with a magnitude of between 1 and 10. 
        # Using values outside of these approximate ranges may result in unpredictable behaviour
        # p.random_distortion(probability=1.0, grid_width=9, grid_height=9, magnitude=9)
        p.random_distortion(probability=1.0, grid_width=3, grid_height=3, magnitude=3)

        # p.crop_random(probability=1, percentage_area=0.5)


        ## Execute and Sample From the Pipeline
        # p.sample(50)
        p.sample(8000)



    def create_tfrecord(self, image_paths, labels, idx_start, idx_end, output_path):

        # Open a TFRecordWriter for the output-file.
        with tf.python_io.TFRecordWriter(output_path) as writer:

            for i in range(idx_start, idx_end):

                utils.print_progress(count=i, total=(idx_end-idx_start))

                image_path = image_paths[i]
                label = labels[i]

                # TODO: Do center cropping
                # img = cv2.imread(image_paths[i])
                # img = cv2.resize(img, (224, 224))


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


                ## Color constancy
                # img = utils_image.color_constancy(img, power=6, gamma=None)
                # if output_path is not None:
                #     img_path_name = os.path.join(os.path.dirname(output_path), os.path.basename(image_path))
                #     img_path_name = img_path_name.split('.')[0] + '_ilu.' + img_path_name.split('.')[1]

                #     # utils_image.save_image(img, img_path_name)
                #     img_save = Image.fromarray(img.astype('uint8'))
                #     img_save.save(img_path_name)




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

