import numpy as np
import tensorflow as tf
import pudb

from tensorflow.examples.tutorials.mnist import input_data

class DataGeneratorMnist:
    def __init__(self, config):
        self.config = config

        ## Load data here
        # self.input = np.ones((500, 784))
        # self.y = np.ones((500, 10))


        # We know that MNIST images are 28 pixels in each dimension.
        img_size = 28

        # Images are stored in one-dimensional arrays of this length.
        img_size_flat = img_size * img_size

        # Tuple with height and width of images used to reshape arrays.
        img_shape = (img_size, img_size)

        # Number of colour channels for the images: 1 channel for gray-scale.
        num_channels = 1

        # Number of classes, one class for each of 10 digits.
        num_classes = 10

        data_mnist = input_data.read_data_sets('temp/mnist/', one_hot=True)

        data_mnist.train.cls = np.argmax(data_mnist.train.labels, axis=1)
        data_mnist.test.cls = np.argmax(data_mnist.test.labels, axis=1)

        self.train = data_mnist.train
        self.test = data_mnist.test

        print("Size of:")
        print("- Training-set:\t\t{}".format(len(data_mnist.train.labels)))
        print("- Test-set:\t\t{}".format(len(data_mnist.test.labels)))
        print("- Validation-set:\t{}".format(len(data_mnist.validation.labels)))



    def next_batch(self, batch_size):
        pass
        # idx = np.random.choice(500, batch_size)
        # yield self.input[idx], self.y[idx]



if __name__ == '__main__':
    dataset_generator_mnist = DataGeneratorMnist(config='')
