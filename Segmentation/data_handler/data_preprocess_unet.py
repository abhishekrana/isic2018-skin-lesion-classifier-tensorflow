##########################################################################################
# Class for preprocessing images
#
# https://challenge2018.isic-archive.com/task1/
# Images eg:
# image_shape = (2016, 3024, 3)
##########################################################################################

import os
os.sys.path.append('./')

import tensorflow as tf
import numpy as np
from PIL import Image
import random
import pudb

import utils.utils as utils
import utils.utils_image as utils_image
from utils.config import process_config

class DataPreprocessUnet():

    def __init__(self, config):

        self.config = config

        random.seed(7)

        image_paths_train, image_paths_test = self.read_dataset(config.dataset_path_train)

        ## For debugging
        image_paths_train = image_paths_train[0:10]
        image_paths_test = image_paths_test[0:10]

        ## Train image: ISIC2018_Task1-2_Training_Input/ISIC_0000000.jpg
        ## Ground Truth train image: ISIC2018_Task1_Training_GroundTruth/ISIC_0000000_segmentation.png
        gt_image_paths_train = []
        for image_path_name in image_paths_train:
            image_name = os.path.basename(image_path_name).split('.')[0] + '_segmentation.png'
            gt_image_paths_train.append(os.path.join(self.config.dataset_path_train_gt, image_name))

        gt_image_paths_test = []
        for image_path_name in image_paths_test:
            image_name = os.path.basename(image_path_name).split('.')[0] + '_segmentation.png'
            ## TODO: _aSk Currently test images are also in train images path
            # gt_image_paths_test.append(os.path.join(self.config.dataset_path_test_gt, image_name))
            gt_image_paths_test.append(os.path.join(self.config.dataset_path_train_gt, image_name))

        self.preprocess_images(image_paths_train, image_paths_test,
                               gt_image_paths_train, gt_image_paths_test)


    def read_dataset(self, dataset_path):
        gt_image_paths_list = []

        ## Read training images
        image_paths = utils_image.get_images_path_list_from_dir(dataset_path, img_format='jpg')
        random.shuffle(image_paths)

        ## Split training images into train and test (currently no test set available)
        image_count = len(image_paths)
        # split = int(image_count*0.8)
        split = 2000

        image_paths_train = image_paths[0:split]
        image_paths_test = image_paths[split:]

        print('Train images: ', len(image_paths_train))
        print('Test images: ', len(image_paths_test))

        return image_paths_train, image_paths_test

    def preprocess_image(self, image_path_name, output_path, image_size):
        ## Load and resize image
        image_np = utils_image.load_image(image_path_name, (image_size))

        ## Save processed image
        image_name = os.path.basename(image_path_name)
        image_name = image_name.split('.')[0] + '_processed.' + image_name.split('.')[-1]
        image_path_name_save = os.path.join(output_path, image_name)

        utils_image.save_image(image_np, image_path_name_save)



    def preprocess_images(self, image_paths_train, image_paths_test, gt_image_paths_train, gt_image_paths_test):
        # TODO: Image size reduced from 3MB to 400kB even if image not changed. Why?

        ## Train
        count = 0
        print('\nPreprocessing Train images')
        for image_path_name in image_paths_train:
            utils.print_progress(count, len(image_paths_train))
            self.preprocess_image(image_path_name, self.config.dataset_path_train_processed,
                                  (self.config.tfr_image_width, self.config.tfr_image_height))
            count = count + 1

        ## Test
        print('\nPreprocessing Test images')
        count = 0
        for image_path_name in image_paths_test:
            utils.print_progress(count, len(image_paths_train))
            self.preprocess_image(image_path_name, self.config.dataset_path_test_processed,
                                  (self.config.tfr_image_width, self.config.tfr_image_height))
            count = count + 1

        ## GT Train
        count = 0
        print('\nPreprocessing Ground Truth Train images')
        for image_path_name in gt_image_paths_train:
            utils.print_progress(count, len(image_paths_train))
            self.preprocess_image(image_path_name, self.config.dataset_path_train_processed,
                                  (self.config.tfr_gt_image_width, self.config.tfr_gt_image_height))
            count = count + 1

        ## GT Test
        print('\nPreprocessing Ground Truth Test images')
        count = 0
        for image_path_name in gt_image_paths_test:
            utils.print_progress(count, len(image_paths_train))
            self.preprocess_image(image_path_name, self.config.dataset_path_test_processed,
                                  (self.config.tfr_gt_image_width, self.config.tfr_gt_image_height))
            count = count + 1

        print('\n')




### MAIN ###
if __name__ == '__main__':
    args = utils.get_args()
    config = process_config(args.config)

    # class Config:
    #     def __init__(self):
    #         self.dataset_path_train = "/mnt/cvpr_shared/abhishek/Practical_2/Dataset/ISIC2018/ISIC2018_Task1-2_Training_Input"
    # config = Config()

    data_preprocess = DataPreprocessUnet(config)


