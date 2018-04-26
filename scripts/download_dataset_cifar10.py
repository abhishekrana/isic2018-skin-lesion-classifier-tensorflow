"""
http://www.cs.toronto.edu/~kriz/cifar.html
"""

import numpy as np
import urllib.request
import os
import tarfile
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import shutil
import glob
import sys

cifar10_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
output_path = os.path.join('..', 'datasets', 'cifar10')
dataset_name = 'cifar-10-batches-py'

def load_image(filename):
    img = Image.open(filename)
    img_arr = np.array(img)
    return img_arr

def load_images_from_dir(dir_path, img_format):
    img_regex = os.path.join(dir_path, '*.' + img_format)
    img_paths = glob.glob(img_regex)
    imgs = [load_image(img_path) for img_path in img_paths]
    return np.array(imgs), img_paths

def save_image(img_arr, img_path, img_name=None):
    img = Image.fromarray(img_arr)
    if img_name is None:
        img.save(img_path)
    else:
        img.save(os.path.join(img_path, img_name))


class Cifar10:

    def __init__(self):
        self.dataset_url = cifar10_url
        self.output_path = output_path
        self.output_file_name = os.path.basename(cifar10_url)
        self.output_file_path_name = os.path.join(self.output_path, self.output_file_name)
        self.dataset_name = dataset_name
        self.dataset_path_name = os.path.join(output_path, dataset_name)

        self.train_images_path = os.path.join(output_path, 'images', 'train')
        self.test_images_path = os.path.join(output_path, 'images', 'test')

    def download_and_extract_dataset(self):

        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.makedirs(self.output_path, exist_ok=True)

        print('Downloading dataset...')
        urllib.request.urlretrieve(url=self.dataset_url, filename=self.output_file_path_name)

        print('Extracting dataset...')
        tar = tarfile.open(self.output_file_path_name)
        tar.extractall(self.output_path)
        tar.close()

        print('Dataset extracted at {}'.format(self.output_path))

    def save_dataset(self, label_names_dict):
        """
        data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
        labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.
        train_dataset: {'data', 'filenames', 'labels'}
        test_dataset: {'data', 'filenames', 'labels'}
        metadata: {'label_names', 'num_cases_per_batch', 'num_vis'}
        """

        # Create directory structure
        if os.path.exists(self.train_images_path):
            shutil.rmtree(self.train_images_path)
        os.makedirs(self.train_images_path, exist_ok=True)
        if os.path.exists(self.test_images_path):
            shutil.rmtree(self.test_images_path)
        os.makedirs(self.test_images_path, exist_ok=True)

        for label_index, label_name in label_names_dict.items():
            os.makedirs(os.path.join(self.train_images_path, label_name), exist_ok=True)
            os.makedirs(os.path.join(self.test_images_path, label_name), exist_ok=True)
            print('label_name', label_name)

        print('Processing train set...')
        train_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        for train_file in train_files:
            with open(os.path.join(self.dataset_path_name, train_file), 'rb') as f:
                dataset = pickle.load(f, encoding='bytes')
                for data, filename, label in zip(dataset[b'data'], dataset[b'filenames'], dataset[b'labels']):
                    # Consecutive 1024 entries store color channels of 32x32 image
                    img_R = data[0:1024].reshape((32, 32))
                    img_G = data[1024:2048].reshape((32, 32))
                    img_B = data[2048:3072].reshape((32, 32))
                    img_np = np.dstack((img_R, img_G, img_B))

                    # Save image
                    filename = filename.decode('utf-8')
                    # filename = filename.split('.')[0] + '_cls_' + str(label) + '.' + filename.split('.')[1]
                    # save_image(img_np, self.train_images_path, filename)
                    save_image(img_np, os.path.join(self.train_images_path, label_names_dict[int(label)], filename))

        print('Train dataset images extracted at ', self.train_images_path)

        print('Processing test set...')
        test_file = 'test_batch'
        with open(os.path.join(self.dataset_path_name, test_file), 'rb') as f:
            dataset = pickle.load(f, encoding='bytes')
            for data, filename, label in zip(dataset[b'data'], dataset[b'filenames'], dataset[b'labels']):
                # Consecutive 1024 entries store color channels of 32x32 image
                img_R = data[0:1024].reshape((32, 32))
                img_G = data[1024:2048].reshape((32, 32))
                img_B = data[2048:3072].reshape((32, 32))
                img_np = np.dstack((img_R, img_G, img_B))

                # Save image
                filename = filename.decode('utf-8')
                # filename = filename.split('.')[0] + '_cls_' + str(label) + '.' + filename.split('.')[1]
                # save_image(img_np, self.test_images_path, filename)
                save_image(img_np, os.path.join(self.test_images_path, label_names_dict[int(label)], filename))
        print('Test dataset images extracted at ', self.test_images_path)

    def load_dataset(self):
        """
        :return:
        train_dataset: {'data', 'filenames', 'labels'}
        test_dataset: {'data', 'filenames', 'labels'}
        metadata: {'label_names', 'num_cases_per_batch', 'num_vis'}
        """
        label_names = []
        with open(os.path.join(self.dataset_path_name, 'batches.meta'), 'rb') as f:
            dataset = pickle.load(f, encoding='bytes')
        label_names_dict = {}
        for idx, label_name in enumerate(dataset[b'label_names']):
            label_names.append(label_name.decode('utf-8'))
            label_names_dict[idx] = label_name.decode('utf-8')
        num_cases_per_batch = dataset[b'num_cases_per_batch']
        num_vis = dataset[b'num_vis']
        metadata = {'label_names': label_names, 'num_cases_per_batch': num_cases_per_batch, 'num_vis': num_vis}
        print('metadata', metadata)

        print('Loading train dataset...')
        train_labels = []
        train_images, train_images_path = load_images_from_dir(self.train_images_path, 'png')
        for image_path in train_images_path:
            train_labels.append(image_path.split('_cls_')[-1].split('.')[0])

        print('Loading test dataset...')
        test_labels = []
        test_images, test_images_path = load_images_from_dir(self.test_images_path, 'png')
        for image_path in test_images_path:
            test_labels.append(image_path.split('_cls_')[-1].split('.')[0])

        print('Dataset loaded')
        return train_images, train_labels, test_images, test_labels, label_names_dict


if __name__ == '__main__':
    print('Start')
    cifar10 = Cifar10()

    cifar10.download_and_extract_dataset()
    train_images, train_labels, test_images, test_labels, label_names_dict = cifar10.load_dataset()
    cifar10.save_dataset(label_names_dict)

    #train_images, train_labels, test_images, test_labels, label_names_dict = cifar10.load_dataset()
    print('Done')
