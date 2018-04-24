##########################################################################################
# Utility functions
##########################################################################################
import argparse
import sys
import os
import shutil
import numpy as np

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args


def print_progress(count, total):
    # Percentage completion.
    pct_complete = float(count) / total

    # Status-message.
    # Note the \r which means the line should overwrite itself.
    msg = "\r- Progress: {0:.1%} [{1}/{2}]".format(pct_complete, count, total)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()



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
            if not os.path.exists(dir_):
                os.makedirs(dir_)
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

