import numpy as np
import urllib.request
import os
import tarfile
import pickle
from PIL import Image
import shutil
import glob
import sys
import csv
from sklearn.model_selection import StratifiedKFold

import utils.utils_image as utils_image
from utils.config import process_config

class Densenet:

    def __init__(self, config, labels_val_name_dict, output_path, dataset_name):
        self.config = config
        self.labels_val_name_dict = labels_val_name_dict
        self.output_path = output_path
        self.dataset_name = dataset_name
        self.dataset_path_name = os.path.join(output_path, dataset_name)
        self.train_images_path = os.path.join(self.dataset_path_name, 'train')
        self.val_images_path = os.path.join(self.dataset_path_name, 'val')
        self.test_images_path = os.path.join(self.dataset_path_name, 'test')

        self.split_train = 8000
        self.split_val   = 1015
        self.split_test  = 1000

        print('dataset_path_name', self.dataset_path_name)
        print('val_images_path', self.val_images_path)
        print('test_images_path', self.test_images_path)


    def save_dataset(self, input_dataset_path, input_dataset_labels_file_path):

        # Create directory structure
        if os.path.exists(self.train_images_path):
            shutil.rmtree(self.train_images_path)
        os.makedirs(self.train_images_path, exist_ok=True)
        if os.path.exists(self.val_images_path):
            shutil.rmtree(self.val_images_path)
        os.makedirs(self.val_images_path, exist_ok=True)
        if os.path.exists(self.test_images_path):
            shutil.rmtree(self.test_images_path)
        os.makedirs(self.test_images_path, exist_ok=True)

        for label_index, label_name in self.labels_val_name_dict.items():
            os.makedirs(os.path.join(self.train_images_path, label_name), exist_ok=True)
            os.makedirs(os.path.join(self.val_images_path, label_name), exist_ok=True)
            os.makedirs(os.path.join(self.test_images_path, label_name), exist_ok=True)


        ## Read cvs to generate lists of corresponding images_name and images_label 
        images_name = []
        images_label = []
        with open(input_dataset_labels_file_path) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for index, row in enumerate(readCSV):

                ## Skip header
                if index == 0:
                    continue

                images_name.append(row[0])

                label_one_hot_encoding = [int(round(float(row[i+1]), 0)) for i in range(7)] 
                images_label.append(np.argmax(label_one_hot_encoding))

                # print('label_one_hot_encoding', np.array(label_one_hot_encoding))
                # print('label', label)


        ## Shuffle dataset
        images_name, images_label = self.shuffle_data(images_name, images_label)
        for i in range(10):
            print('images_name:{}, images_label:{}'.format(images_name[i], images_label[i]))


        ## Split dataset into train, val and test
        images_name_train = images_name[0:self.split_train]
        images_label_train = images_label[0:self.split_train]

        images_name_val = images_name[self.split_train:self.split_train+self.split_val]
        images_label_val = images_label[self.split_train:self.split_train+self.split_val]

        images_name_test = images_name[self.split_train+self.split_val:self.split_train+self.split_val+self.split_test]
        images_label_test = images_label[self.split_train+self.split_val:self.split_train+self.split_val+self.split_test]


        print('\nData Distribution:')
        self.display_data_distribution(images_label)

        print('\nData Distribution Train:')
        self.display_data_distribution(images_label_train)

        print('\nData Distribution Val:')
        self.display_data_distribution(images_label_val)

        print('\nData Distribution Test:')
        self.display_data_distribution(images_label_test)


        ## Save images
        print('\nSaving images...')
        for image_name_train, image_label_train in zip(images_name_train, images_label_train):
            shutil.copy2(os.path.join(input_dataset_path, image_name_train + '.jpg'), os.path.join(self.train_images_path, self.labels_val_name_dict[image_label_train]))

        for image_name_val, image_label_val in zip(images_name_val, images_label_val):
            shutil.copy2(os.path.join(input_dataset_path, image_name_val + '.jpg'), os.path.join(self.val_images_path, self.labels_val_name_dict[image_label_val]))

        for image_name_test, image_label_test in zip(images_name_test, images_label_test):
            shutil.copy2(os.path.join(input_dataset_path, image_name_test + '.jpg'), os.path.join(self.test_images_path, self.labels_val_name_dict[image_label_test]))



    def display_data_distribution(self, images_labels):
        print('Total:', len(images_labels))
        data_dist = dict()
        for i in images_labels:
            data_dist[i] = data_dist.get(i, 0) + 1

        # print('data_dist', data_dist)
        for label_idx, label_freq in data_dist.items():
            print('{:10s}:{:10d} : {:10.1f}%'.format(self.labels_val_name_dict[label_idx], label_freq, 
                                                    (label_freq/len(images_labels)*100)))


    def shuffle_data(self, list1, list2):
        """Joint shuffling of the lists"""
        permutation = np.random.permutation(len(list1))
        list1_shuffled = []
        list2_shuffled = []
        for i in permutation:
            list1_shuffled.append(list1[i])
            list2_shuffled.append(list2[i])
        return list1_shuffled, list2_shuffled


if __name__ == '__main__':

    try:
        args = utils.get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        config_file = 'configs/config_densenet.json'
        config = process_config(config_file)


    np.random.seed(config.seed)
    print('seed: {}'.format(config.seed))

    input_dataset_path = '/usr/data/cvpr_shared/abhishek/Practical_2/Dataset/ISIC2018/Classification/ISIC2018_Task3_Training_Input'
    input_dataset_labels_file_path = '/usr/data/cvpr_shared/abhishek/Practical_2/Dataset/ISIC2018/Classification/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv'
    output_path = os.path.join('..', 'datasets', 'densenet')

    ## Reverse key and values of dict
    labels_val_name_dict = {v: k for k, v in config.labels.items()}
    # {0: 'MEL', 1: 'NV', 2: 'BCC', 3: 'AKIEC', 4: 'BKL', 5: 'DF', 6: 'VASC'}
    print('labels_val_name_dict', labels_val_name_dict)

    densenet = Densenet(config, labels_val_name_dict, output_path='datasets', dataset_name='densenet')
    densenet.save_dataset(input_dataset_path, input_dataset_labels_file_path)

    print('Done')



