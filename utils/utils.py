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


def shuffle_data(list1, list2):
    """Joint shuffling of the lists"""
    permutation = np.random.permutation(len(list1))
    list1_shuffled = []
    list2_shuffled = []
    for i in permutation:
        list1_shuffled.append(list1[i])
        list2_shuffled.append(list2[i])
    return list1_shuffled, list2_shuffled


def print_progress(count, total):
    # Percentage completion.
    pct_complete = float(count) / total

    # Status-message.
    # Note the \r which means the line should overwrite itself.
    msg = "\r- Progress: {0:.1%} [{1}/{2}]".format(pct_complete, count, total)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


