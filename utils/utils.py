##########################################################################################
# Utility functions
##########################################################################################
import argparse
import sys
import os
import shutil
import numpy as np
import datetime
import logging
import signal
import glob
import tensorflow as tf
import sklearn

import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config_file',
        metavar='C',
        default='None',
        help='The Configuration file')
    argparser.add_argument(
        '-m', '--mode',
        metavar='M',
        default='None',
        help='tfr/train/eval/test')
    args = argparser.parse_args()
    return args


def logger_init(config, log_level):
    """Initialize logger

    Arguments:
        config:
            Configuration
        mode:
            train, test, val
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

    log_file_path = os.path.join(config.output_path, config.exp_name, config.mode.lower())
    log_file_path_name = os.path.join(log_file_path, 'log_' + timestamp)

    os.makedirs(log_file_path, exist_ok=True)

    # logFormatter = logging.Formatter("%(asctime)s | %(filename)20s:%(lineno)s | %(funcName)20s() | %(threadName)-12.12s | %(levelname)-5.5s | %(message)s")
    logFormatter = logging.Formatter("%(asctime)s | %(filename)20s:%(lineno)s | %(funcName)20s() | %(message)s")

    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler(log_file_path_name)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    rootLogger.setLevel(log_level)


def save_model(config):
    """Save model

    Arguments:
        config:
            Configuration
    """
    timestamp_end = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

    output_name = '{}_output'.format(timestamp_end)
    logging.debug('output_name {}'.format(output_name))

    shutil.move(config.output_path, output_name)
    logging.debug('Model Saved {}', output_name)


def save_code(config):

    for code_dir in ['base', 'configs', 'data_handler', 'datasets', 'mains', 'models', 'scripts', 'trainers', 'utils']:
        output_code_path = os.path.join(config.output_path, config.exp_name, 'code', code_dir)
        os.makedirs(output_code_path, exist_ok=True)
        for file_type in ['*.py', '*.json', '*.sh']:
            files = glob.iglob(os.path.join(code_dir, file_type))
            for f in files:
                if os.path.isfile(f):
                    shutil.copy2(f, output_code_path)


def signal_handler(config):
    """Signal Handler for Ctrl+C

    Arguments:
        config:
            Configuration
    """
    def sig_hdlr(*args):
        text = input("Save Model?")
        if text == 'y' or text == 'Y':
            save_model(config)
            exit(0)
        else:
            timestamp_end = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
            logging.debug('timestamp_end {}'.format(timestamp_end))
            exit(0)

    signal.signal(signal.SIGINT, sig_hdlr)


def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    NOTE: Input should be a list
    """
    if type(dirs) is not list:
        print('Input is not a list')
        exit(-1)

    try:
        for dir_ in dirs:
            os.makedirs(dir_, exist_ok=True)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def remove_dirs(dirs):
    """
    WARNING: Input should be a list otherwise it will delete /
    """
    if type(dirs) is not list:
        print('Input is not a list')
        exit(-1)

    try:
        for dir_ in dirs:
            if os.path.exists(dir_):
                print('Deleting {}'.format(dir_))
                shutil.rmtree(dir_)
        return 0
    except Exception as err:
        print("Removing directories error: {0}".format(err))
        exit(-1)


def one_hot_encoded(class_numbers, num_classes=None):
    """
    Generate the One-Hot encoded class-labels from an array of integers.

    For example, if class_number=2 and num_classes=4 then
    the one-hot encoded label is the float array: [0. 0. 1. 0.]

    :param class_numbers:
        Array of integers with class-numbers.
        Assume the integers are from zero to num_classes-1 inclusive.

    :param num_classes:
        Number of classes. If None then use max(class_numbers)+1.

    :return:
        2-dim array of shape: [len(class_numbers), num_classes]
    """

    # Find the number of classes if None is provided.
    # Assumes the lowest class-number is zero.
    if num_classes is None:
        num_classes = np.max(class_numbers) + 1

    return np.eye(num_classes, dtype=float)[class_numbers]


def shuffle_data_2(list1, list2):
    """Joint shuffling of the lists"""
    permutation = np.random.permutation(len(list1))
    list1_shuffled = []
    list2_shuffled = []
    for i in permutation:
        list1_shuffled.append(list1[i])
        list2_shuffled.append(list2[i])
    return list1_shuffled, list2_shuffled

def shuffle_data(a, b):
    a = np.array(a)
    b = np.array(b)
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def print_progress(count, total):
    # Percentage completion.
    pct_complete = float(count) / total

    # Status-message.
    # Note the \r which means the line should overwrite itself.
    msg = "\r- Progress: {0:.1%} [{1}/{2}]".format(pct_complete, count, total)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()

def get_config_class_weight(config, data):

    ## 10 sec wasted here; Hence hardcoding for 8000 train images; for categorical labels
    if (config.debug_train_images_count == 8000) or (config.debug_train_images_count == 0):
        class_weight =  [ 1.2755102,   0.21409838,  2.72108844,  4.53514739,  1.29282482, 12.84109149, 9.44510035]
    else:
        filenames_regex = os.path.join(config.tfrecords_path_train, '*.tfr')
        filenames = glob.glob(filenames_regex)
        if not filenames:
            logging.debug('ERROR: No .tfr files found')
            exit(1)
        data_train_op = data.input_fn(filenames=filenames, train=False, batch_size=config.debug_train_images_count, buffer_size=config.data_gen_buffer_size)

        with tf.Session() as sess:
            data_train = sess.run(data_train_op)
        labels_train = np.argmax(data_train[1], axis=1)
        class_weight = sklearn.utils.class_weight.compute_class_weight('balanced', np.unique(labels_train), labels_train)

    class_weight_dict = {}
    for idx, weight in enumerate(class_weight):
        class_weight_dict[idx] = weight

    config.class_weight = class_weight_dict

    logging.debug('class_weight {}'.format(config.class_weight))



def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def get_confusion_matrix(config, gt_labels, pred_labels):

    plot_path = os.path.join(config.output_path, config.exp_name, 'plots')
    os.makedirs(plot_path, exist_ok=True)

    class_names = []
    for key_val in sorted(config.labels.items(), key=lambda x: x[1]):
        class_names.append(key_val[0])

    ## Compute confusion matrix
    cnf_matrix = confusion_matrix(y_true=gt_labels, y_pred=pred_labels)
    logging.debug('cnf_matrix {}'.format(cnf_matrix))
    # np.set_printoptions(precision=2)

    ## Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')
    plt.savefig(os.path.join(plot_path, 'confusion_matrix.png'))

    ## Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')
    plt.savefig(os.path.join(plot_path, 'confusion_matrix_norm.png'))

    # plt.show()


