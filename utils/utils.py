##########################################################################################
# Utility functions
##########################################################################################
import os
import sys
import shutil
import argparse
import numpy as np
import datetime
import logging
import signal
import glob
import sklearn
import glob
import itertools
import numpy as np
import matplotlib
import pandas as pd
import six
# import matplotlib; matplotlib.pyplot.switch_backend('agg')
import matplotlib.pyplot as plt

from itertools import cycle
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc, recall_score, accuracy_score, precision_score
from sklearn.preprocessing import label_binarize
from scipy import interp



from textwrap import wrap
import re
import tfplot

import tensorflow as tf

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
    argparser.add_argument(
        '-d', '--debug',
        metavar='D',
        default='None',
        help='Debug mode for tf_dbg')

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

def set_config_class_weights(config, data):

    ## 10 sec wasted here; Hence hardcoding for 8000 train images; for categorical labels
    if (config.debug_train_images_count == 8000) or (config.debug_train_images_count == 0):
        class_weights =  np.array([ 1.2755102,   0.21409838,  2.72108844,  4.53514739,  1.29282482, 12.84109149, 9.44510035])
    else:
        # filenames = tf.data.Dataset.list_files(os.path.join(config.tfrecords_path_train, '*.tfr'))
        data_train_op = data.input_fn(
                                file_pattern=os.path.join(config.tfrecords_path_train, '*.tfr'),
                                mode='train', 
                                batch_size=config.debug_train_images_count, 
                                buffer_size=config.data_gen_buffer_size,
                                num_repeat = 1
                                )

        with tf.Session() as sess:
            data_train = sess.run(data_train_op)

        labels_train = np.argmax(data_train[1], axis=1)
        logging.debug('labels_train {}'.format(len(labels_train)))
        class_weights = sklearn.utils.class_weight.compute_class_weight('balanced', np.unique(labels_train), labels_train)

    if len(class_weights) < config.num_classes:
        class_weights = class_weights.tolist() + (config.num_classes-len(class_weights))*[0]


    config.class_weights = class_weights.tolist()
    config.class_weights_dict = dict(enumerate(class_weights))
    logging.debug('class_weights {}'.format(config.class_weights_dict))



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
    plt.xlabel('True label')
    plt.ylabel('Predicted label')


def get_confusion_matrix(config, gt_labels, pred_labels, dataset_split_name):

    plot_path = os.path.join(config.output_path, config.exp_name, 'plots', dataset_split_name)
    os.makedirs(plot_path, exist_ok=True)

    class_names = []
    for key_val in sorted(config.labels.items(), key=lambda x: x[1]):
        class_names.append(key_val[0])

    ## Compute confusion matrix
    cnf_matrix = confusion_matrix(y_true=gt_labels, y_pred=pred_labels)

    # Swap axis
    cnf_matrix = np.transpose(cnf_matrix)
    logging.debug('cnf_matrix \n{}'.format(cnf_matrix))
    np.set_printoptions(precision=2)

    cnf = np.transpose(cnf_matrix)

    ## Plot non-normalized confusion matrix
    # plt.figure(figsize=(5,5))
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')
    plt.savefig(os.path.join(plot_path, 'confusion_matrix.png'))
    plt.close()

    ## Plot normalized confusion matrix
    # plt.figure(figsize=(5,5))
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')
    plt.savefig(os.path.join(plot_path, 'confusion_matrix_norm.png'))
    plt.close()

    # plt.show()


def get_files_from_pattern(file_path_pattern):
    file_path_names = glob.glob(file_path_pattern)
    if not file_path_names:
        logging.error('ERROR: No files found')
        exit(1)

    return file_path_names



def summary_confusion_matrix(config, correct_labels, predict_labels, labels, title='Confusion matrix', tensor_name = 'MyFigure/image', normalize=False):
    ''' 
    Parameters:
        correct_labels                  : These are your true classification categories.
        predict_labels                  : These are you predicted classification categories
        labels                          : This is a lit of labels which will be used to display the axix labels
        title='Confusion matrix'        : Title for your matrix
        tensor_name = 'MyFigure/image'  : Name for the output summay tensor

    Returns:
        summary: TensorFlow summary 

    Other itema to note:
        - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc. 
        - Currently, some of the ticks dont line up due to rotations.
    '''

    labels_map_inv = {v: k for k, v in config.labels.items()}
    correct_labels = [labels_map_inv[label_idx] for label_idx in correct_labels]
    predict_labels = [labels_map_inv[label_idx] for label_idx in predict_labels]

    class_names = []
    for key_val in sorted(config.labels.items(), key=lambda x: x[1]):
        class_names.append(key_val[0])

    cm = confusion_matrix(y_true=correct_labels, y_pred=predict_labels, labels=class_names)
    if normalize:
        cm = cm.astype('float')*10 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = cm.astype('int')

    np.set_printoptions(precision=2)
    #fig, ax = matplotlib.figure.Figure()

    # fig = matplotlib.figure.Figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
    fig = matplotlib.figure.Figure(figsize=(10, 10), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=4, rotation=-90,  ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=4, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=6, verticalalignment='center', color= "black")
    fig.set_tight_layout(True)
    summary = tfplot.figure.to_summary(fig, tag=tensor_name)
    
    return summary



# Plot an ROC. pred - the predictions, y - the expected output.
# def plot_roc(y_pred, y_true, pos_label):
#     fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label)
#     roc_auc = auc(fpr, tpr)

#     plt.figure()
#     plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic (ROC)')
#     plt.legend(loc="lower right")
#     plt.show()

def summary_roc(config, correct_labels, predict_labels, labels, title='Confusion matrix', tensor_name = 'MyFigure/image', normalize=False):
    y_true = correct_labels
    y_pred = predict_labels

    # y_true = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2]
    # y_pred = [0,0,0,0,0,1,1,1,0,0,1,1,1,2,1,1,2,2,2,2,2,2,2,2,2,2,2]

    # class_names = ['cats', 'dogs', 'rabbits']
    # class_names_2 = ['cats', 'dogs']
    # y_true = [0,0,1,1,2,2]
    # y_pred = [0,0,1,1,1,2]

    classes = np.unique(np.array(y_true))
    for cls in classes:
        
        print('Class:', cls)

        y_true_class = [1 if x==cls else 0 for x in y_true]
        print(y_true_class)
        
        y_pred_class = [1 if x==cls else 0 for x in y_pred]
        print(y_pred_class)
        
        # plot_roc(y_pred_class, y_true_class, pos_label=None)
        fpr, tpr, _ = roc_curve(y_true_class, y_pred_class)
        roc_auc = auc(fpr, tpr)


        
        #metric_confusion_matrix(y_true, y_pred, class_names)
        # metric_confusion_matrix(y_true_class, y_pred_class, [class_names[cls]] + ['rest'])

        np.set_printoptions(precision=2)
        #fig, ax = matplotlib.figure.Figure()

        # fig = matplotlib.figure.Figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
        fig = matplotlib.figure.Figure(figsize=(10, 10), dpi=320, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)


        # labels_map_inv = {v: k for k, v in config.labels.items()}
        # correct_labels = [labels_map_inv[label_idx] for label_idx in correct_labels]
        # predict_labels = [labels_map_inv[label_idx] for label_idx in predict_labels]
        # class_names = []
        # for key_val in sorted(config.labels.items(), key=lambda x: x[1]):
        #     class_names.append(key_val[0])
        # cm = confusion_matrix(y_true=correct_labels, y_pred=predict_labels, labels=class_names)
        # im = ax.imshow(cm, cmap='Oranges')

        im = plt.plot(fpr, tpr)

        classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
        classes = ['\n'.join(wrap(l, 40)) for l in classes]

        tick_marks = np.arange(len(classes))

        ax.set_xlabel('Predicted', fontsize=7)
        ax.set_xticks(tick_marks)
        c = ax.set_xticklabels(classes, fontsize=4, rotation=-90,  ha='center')
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.tick_bottom()

        ax.set_ylabel('True Label', fontsize=7)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, fontsize=4, va ='center')
        ax.yaxis.set_label_position('left')
        ax.yaxis.tick_left()

        # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        #     ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=6, verticalalignment='center', color= "black")
        fig.set_tight_layout(True)
        summary = tfplot.figure.to_summary(fig, tag=tensor_name)
        
        return summary

def summary_pr_fscore(config, correct_labels, predict_labels, labels):

    # labels_map_inv = {v: k for k, v in config.labels.items()}
    # correct_labels = [labels_map_inv[label_idx] for label_idx in correct_labels]
    # predict_labels = [labels_map_inv[label_idx] for label_idx in predict_labels]

    class_names = []
    for key_val in sorted(config.labels.items(), key=lambda x: x[1]):
        class_names.append(key_val[0])

    # precision, recall, fscore, support = precision_recall_fscore_support(y_true=correct_labels, y_pred=predict_labels, labels=class_names, average=None)
    precision, recall, fscore, support = precision_recall_fscore_support(y_true=correct_labels, y_pred=predict_labels, average=None)

    logging.debug('{:10.10}  {:12.12}  {:10.10} {:10.10}{:10.10}'.format('Class', 'Precision', 'Recall', 'FScore', 'Support'))
    for l, p, r, f, s in zip(class_names, precision, recall, fscore, support):
        logging.debug('{:10} {:10.2} {:10.2} {:10.2} {:10}'.format(l, p, r, f, s))

    print('{:10.10}  {:12.12}  {:10.10} {:10.10}{:10.10}'.format('Class', 'Precision', 'Recall', 'FScore', 'Support'))
    for l, p, r, f, s in zip(class_names, precision, recall, fscore, support):
        print('{:10} {:10.2} {:10.2} {:10.2} {:10}'.format(l, p, r, f, s))


    precision, recall, fscore, support = precision_recall_fscore_support(y_true=correct_labels, y_pred=predict_labels, labels=class_names, average='micro')
    logging.debug('\nMicro    - Precision:{} Recall:{} fscore:{} Support:{}'.format(precision, recall, fscore, support))

    precision, recall, fscore, support = precision_recall_fscore_support(y_true=correct_labels, y_pred=predict_labels, labels=class_names, average='macro')
    logging.debug('\nMacro    - Precision:{} Recall:{} fscore:{} Support:{}'.format(precision, recall, fscore, support))

    precision, recall, fscore, support = precision_recall_fscore_support(y_true=correct_labels, y_pred=predict_labels, labels=class_names, average='weighted')
    logging.debug('\nWeighted - Precision:{} Recall:{} fscore:{} Support:{}'.format(precision, recall, fscore, support))


def debug_tf(output, *args):
    def _debug_func(*args):
        pu.db
        return False

    # debug_op = tf.py_func(func=_debug_func, inp=[logits, class_weights, weights, unweighted_losses, weighted_losses, loss], Tout=[tf.bool])
    debug_op = tf.py_func(func=_debug_func, inp=[*args], Tout=[tf.bool])
    with tf.control_dependencies(debug_op):
        output = tf.identity(output, name='out')

    return output

def get_metrics(config, gt_labels, pred_labels, dataset_split_name):
    """
    """
    class_names = []
    for key_val in sorted(config.labels.items(), key=lambda x: x[1]):
        class_names.append(key_val[0])

    ## Compute confusion matrix
    cnf_matrix = confusion_matrix(y_true=gt_labels, y_pred=pred_labels)
    logging.debug('cnf_matrix \n{}'.format(cnf_matrix))
    np.set_printoptions(precision=2)

    # Swap axis so that Left:Pred, Top:True
    cnf_matrix = np.transpose(cnf_matrix)
    logging.debug('Pred:Left; True:Top')
    logging.debug('cnf_matrix \n{}'.format(cnf_matrix))

    round_val = 2

    metrics = []
    for idx in range(cnf_matrix.shape[0]):
        PPV = 0.0
        TPR = 0.0
        TNR = 0.0
        NPV = 0.0
        F1 = 0.0
        accuracy = 0.0
        
        # Power
        TP = cnf_matrix[idx, idx]
        
        # Type I Error
        FP = np.sum(cnf_matrix[idx, :]) - TP
        
        # Type II Error
        FN = np.sum(cnf_matrix[:, idx]) - TP
        
        TN = np.sum(cnf_matrix) - TP - FP - FN
        
        # PPV/Precision
        if TP != 0:
            PPV = TP/(TP+FP)
        
        # TPR/Recall/Sensitivity
        if TP != 0:
            TPR = TP/(TP+FN)
        
        # TNR/Specificity
        if TN != 0:
            TNR = TN/(TN+FP)
        
        if TN != 0:
            NPV = TN/(TN+FN)
        

        if (TP + TN) != 0:
            accuracy = (TP + TN)/(TP + TN + FP + FN)
        
        if TP != 0:
            F1 = (2*TP)/((2*TP) + FP + FN)
        
        # print('\nClass{}: {}'.format(idx, class_names[idx]))
        # print('TP:{}, FP:{}, FN:{}, TN:{}'.format(TP, FP, FN, TN))
        # print('PPV/Precision:{:13}'.format(np.round(PPV, round_val)))
        # print('TPR/Recall/Sensitivity:{:4}'.format(np.round(TPR, round_val)))
        # print('TNR/Specificity:{:12}'.format(np.round(TNR, round_val)))
        # print('NPV:{:25}'.format(np.round(NPV, round_val)))
        # print('Accuracy:{:20}'.format(np.round(accuracy, round_val)))
        # print('F1:{:26}'.format(np.round(F1, round_val)))
        # print('True Samples:{:14}'.format(TP+FN))

        metrics.append([class_names[idx], TP, FP, FN, TN, np.round(PPV, round_val), np.round(TPR, round_val), 
            np.round(TNR, round_val), np.round(NPV, round_val), np.round(accuracy, round_val), 
            np.round(F1, round_val), (TP+FN)])

    print('PPV/Precision, TPR/Recall/Sensitivity, TNR/Specificity, ACC:Accuracy, TS:TrueSamples\n')

    metrics_names = ['CLS', 'TP', 'FP', 'FN', 'TN', 'PPV', 'TPR', 'TNR', 'NPV', 'ACC', 'F1', 'TS']
    metrics_df = pd.DataFrame(metrics, index=None, columns=metrics_names)
    print(metrics_df)

    threshold = 0.5
    plot_path = os.path.join(config.output_path, config.exp_name, 'plots', dataset_split_name)
    img_path_name = os.path.join(plot_path, 'metrics_threshold_' + str(threshold) + '.jpg')
    logging.debug('img_path_name {}'.format(img_path_name))
    os.makedirs(plot_path, exist_ok=True)

    save_dataframe_to_image(metrics_df, img_path_name)




def save_dataframe_to_image(data, img_path_name, col_width=1.2, row_height=0.8, font_size=16,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=1,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])

    plt.savefig(img_path_name)
    plt.close()

    return ax



def adjusted_classes(prob, threshold=0.5):
    return [1 if y >= t else 0 for y in prob]


def gen_precision_recall_curve(config, labels_gt, labels_prob, dataset_split_name):
    
    
    labels_map_inv = {v: k for k, v in config.labels.items()}
    plot_path = os.path.join(config.output_path, config.exp_name, 'plots', dataset_split_name)
    os.makedirs(plot_path, exist_ok=True)

    for class_idx in range(config.num_classes):

        labels_gt_bin = [1 if x==class_idx else 0 for x in labels_gt]
        labels_prob_bin = [x[class_idx] for x in labels_prob]

        average_precision = average_precision_score(labels_gt_bin, labels_prob_bin)
        precision, recall, thresholds = precision_recall_curve(labels_gt_bin, labels_prob_bin)

        for p, r, t in zip(precision, recall, thresholds):
            print('Precision:{:0.2}, Recall:{:0.2}, Thresholds:{:0.2}'.format(p, r, t))

        plt.figure(figsize=(10,10))
        plt.plot(recall, precision)
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('2-class Precision-Recall curve: AP={0:0.2f} [{1}]'.format(average_precision, labels_map_inv[class_idx]))

        img_path_name = os.path.join(plot_path, 'pr_curve_' + labels_map_inv[class_idx] + '.jpg')
        logging.debug('img_path_name {}'.format(img_path_name))

        plt.savefig(img_path_name)
        plt.close()


def gen_roc_curve(config, labels_gt, labels_prob, dataset_split_name):

    labels_gt_one_hot = label_binarize(labels_gt, classes=np.arange(config.num_classes))
    y_score = np.array(labels_prob)
    y_test = labels_gt_one_hot

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    labels_map_inv = {v: k for k, v in config.labels.items()}

    plot_path = os.path.join(config.output_path, config.exp_name, 'plots', dataset_split_name)
    os.makedirs(plot_path, exist_ok=True)

    for i in range(config.num_classes):

        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure(figsize=(10,10))
        plt.plot(fpr[i], tpr[i], color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt_title = 'Receiver operating characteristic [' + labels_map_inv[i] + ']'
        plt.title(plt_title)
        plt.legend(loc="lower right")
        # plt.show()

        img_path_name = os.path.join(plot_path, 'roc_curve_' + labels_map_inv[i] + '.jpg')
        logging.debug('img_path_name {}'.format(img_path_name))

        plt.savefig(img_path_name)
        plt.close()


    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(config.num_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(config.num_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= config.num_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(10,10))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    colors = cycle(['brown', 'burlywood', 'CadetBlue', 'darkorange', 'cornflowerblue', 'aqua', 'red'])
    for i, color in zip(range(config.num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-class')
    plt.legend(loc="lower right")
    # plt.show()

    img_path_name = os.path.join(plot_path, 'roc_curve' + '.jpg')
    logging.debug('img_path_name {}'.format(img_path_name))
    plt.savefig(img_path_name)
    plt.close()



"""

def _debug_func(*args):
    pu.db
    return False
debug_op = tf.py_func(func=_debug_func, inp=[logits, class_weights, weights, unweighted_losses, weighted_losses, loss], Tout=[tf.bool])
with tf.control_dependencies(debug_op):
    loss = tf.identity(loss, name='out')


def _debug_func(
    logits, class_weights, weights, unweighted_losses, weighted_losses, loss
    ):
    pu.db; return False
    debug_op = tf.py_func(func=_debug_func, Tout=[tf.bool], inp=[
        logits, class_weights, weights, unweighted_losses, weighted_losses, loss
    ])
    with tf.control_dependencies(debug_op):
        loss = tf.identity(loss)

"""
