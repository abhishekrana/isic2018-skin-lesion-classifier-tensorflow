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


class Densenet:

    def __init__(self, output_path, dataset_name):
        self.output_path = output_path
        self.dataset_name = dataset_name
        self.dataset_path_name = os.path.join(output_path, dataset_name)
        self.train_images_path = os.path.join(self.dataset_path_name, 'train')
        self.test_images_path = os.path.join(self.dataset_path_name, 'test')
        print('dataset_path_name', self.dataset_path_name)
        print('train_images_path', self.train_images_path)
        print('test_images_path', self.test_images_path)

    def save_dataset(self, label_names_dict, input_dataset_path, input_dataset_labels_file_path):

        # Create directory structure
        if os.path.exists(self.train_images_path):
            shutil.rmtree(self.train_images_path)
        os.makedirs(self.train_images_path, exist_ok=True)
        if os.path.exists(self.test_images_path):
            shutil.rmtree(self.test_images_path)
        os.makedirs(self.test_images_path, exist_ok=True)

        for label_index, label_name in label_names_dict.items():
            print('label_index', label_index)
            print('label_name', label_name)
            os.makedirs(os.path.join(self.train_images_path, label_name), exist_ok=True)
            os.makedirs(os.path.join(self.test_images_path, label_name), exist_ok=True)


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


        images_name, images_label = self.shuffle_data(images_name, images_label)
        print('images_name', images_name[0:10])
        print('images_label', images_label[0:10])


        split = 8000
        images_name_train = images_name[0:split]
        images_label_train = images_label[0:split]
        print('images_name_train', len(images_name_train))
        print('images_label_train', len(images_label_train))

        images_name_test = images_name[split:]
        images_label_test = images_label[split:]
        print('images_name_test', len(images_name_test))
        print('images_label_test', len(images_label_test))

        for image_name_train, image_label_train in zip(images_name_train, images_label_train):
            # print('image_label_train', image_label_train)
            # print('image_name_train', image_name_train)
            shutil.copy2(os.path.join(input_dataset_path, image_name_train + '.jpg'), os.path.join(self.train_images_path, label_names_dict[image_label_train]))

        for image_name_test, image_label_test in zip(images_name_test, images_label_test):
            # print('image_label_test', image_label_test)
            # print('image_name_test', image_name_test)
            shutil.copy2(os.path.join(input_dataset_path, image_name_test + '.jpg'), os.path.join(self.test_images_path, label_names_dict[image_label_test]))


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
    print('Start')

    np.random.seed(7)

    input_dataset_path = '/usr/data/cvpr_shared/abhishek/Practical_2/Dataset/ISIC2018/Classification/ISIC2018_Task3_Training_Input'
    input_dataset_labels_file_path = '/usr/data/cvpr_shared/abhishek/Practical_2/Dataset/ISIC2018/Classification/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv'
    output_path = os.path.join('..', 'datasets', 'densenet')

    densenet = Densenet(output_path='../datasets', dataset_name='densenet')

    label_key_dict = {"MEL": 0, "NV": 1, "BCC": 2, "AKIEC": 3, "BKL": 4, "DF": 5, "VASC": 6}
    label_names_dict = {v: k for k, v in label_key_dict.items()}

    densenet.save_dataset(label_names_dict, input_dataset_path, input_dataset_labels_file_path)

    print('Done')



