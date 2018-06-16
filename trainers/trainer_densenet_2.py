##########################################################################################
# Training
##########################################################################################
import os
os.sys.path.append('./')
os.sys.path.append('../')

import matplotlib; matplotlib.pyplot.switch_backend('agg')

from base.base_trainer import BaseTrain
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import os
import glob
import pudb
import random
from tensorflow.python import debug as tf_debug
import logging
from PIL import Image
import datetime
import keras
import csv
import cv2
from PIL import Image, ImageOps


from data_handler.data_generator_densenet import DataGeneratorDensenet
from data_handler.tfrecords_densenet import TFRecordsDensenet
from models.model_densenet import ModelDensenet
from trainers.trainer_densenet import TrainerDensenet
from utils.config import process_config
# from utils.tf_logger import TFLogger
import utils.utils as utils
import utils.utils_image as utils_image



class TrainerDensenet_2(BaseTrain):

    def __init__(self, sess, model, data, config,logger):
        super(TrainerDensenet_2, self).__init__(sess, model, data, config,logger)

        self.summary_writer = tf.summary.FileWriter(
                                                os.path.join(self.config.summary_dir, self.config.mode),
                                                graph=self.sess.graph, flush_secs=30)
            

    def run_epoch_train(self, mode, epoch):

        assert (mode=='train')

        num_steps = int(self.config.debug_train_images_count/self.config.batch_size)
        logging.debug('num_steps {}'.format(num_steps))

        data_batch = self.data.input_fn(
                                file_pattern=os.path.join(self.config.tfrecords_path_train, '*.tfr'),
                                mode=mode,
                                batch_size=self.config.batch_size,
                                buffer_size=self.config.data_gen_buffer_size
                                )
        
        for step in range(num_steps):

            features_dict, labels_gt = self.sess.run(data_batch)
            features = features_dict[self.config.model_name + '_input']

            _, loss, metrics, summary = self.sess.run([
                        self.model.train_op, 
                        self.model.loss, 
                        self.model.metrics,
                        self.model.summary_op,
                        # self.model.summary_pr_op,
                        # self.model.tf_print_op
                        ],
                    feed_dict={
                        self.model.features: features, 
                        self.model.labels: labels_gt
                        }
                    )

            # global_step refer to the number of batches seen by the graph. When it is passed in the 
            # optimizer.minimize() argument list, the variable is increased by one
            global_step = self.sess.run(tf.train.get_global_step())

            logging.debug('Epoch:{}, global_step:{}, step:{}, loss:{}, accuracy:{}'.format(epoch, global_step, step, loss, metrics['accuracy']))

            ## Save checkpoints
            if (global_step%self.config.train_save_checkpoints_steps) == 0:
                self.model.saver.save(
                        self.sess, 
                        save_path=os.path.join(self.config.checkpoint_dir, 'model_{}.ckpt'.format(global_step))
                        )

            ## Save summary
            if (global_step%self.config.train_save_summary_steps) == 0:
                self.summary_writer.add_summary(summary, global_step)
                # self.summary_writer.add_summary(summary_pr, global_step)



    def train(self):
        epoch = 0
        for epoch in range(self.config.num_epochs):
            self.run_epoch_train(mode='train', epoch=epoch)


    def run_epoch_eval(self, mode, epoch):

        assert (mode=='eval')

        num_steps = int(self.config.debug_val_images_count/self.config.batch_size_eval)
        logging.debug('num_steps {}'.format(num_steps))

        data_batch = self.data.input_fn(
                                file_pattern=os.path.join(self.config.tfrecords_path_val, '*.tfr'),
                                mode=mode,
                                batch_size=self.config.batch_size_eval,
                                buffer_size=self.config.data_gen_buffer_size
                                )
        
        gs_labels_gt = []
        gs_labels_pred = []

        for step in range(num_steps):

            try:
                features_dict, labels_gt = self.sess.run(data_batch)
            except Exception:
                logging.debug('Input data stream read completely {}'.format(Exception))
                # pass
                return

            ## labels_gt is one-hot
            gs_labels_gt += np.argmax(labels_gt, axis=1).tolist()

            features = features_dict[self.config.model_name + '_input']

            loss, metrics, summary, labels_pred_cls = self.sess.run([
                        self.model.loss, 
                        self.model.metrics,
                        self.model.summary_op,
                        self.model.labels_pred_cls
                        # self.model.summary_pr_op,
                        ],
                    feed_dict={
                        self.model.features: features, 
                        self.model.labels: labels_gt
                        }
                    )

            gs_labels_pred += labels_pred_cls.tolist()

            # global_step refer to the number of batches seen by the graph. When it is passed in the 
            # optimizer.minimize() argument list, the variable is increased by one
            global_step = self.sess.run(tf.train.get_global_step())

            logging.debug('Epoch:{}, global_step:{}, step:{}, loss:{}, accuracy:{}'.format(epoch, global_step, step, loss, metrics))

        ## Save summary
        self.summary_writer.add_summary(summary, global_step)
        # self.summary_writer.add_summary(summary_pr, global_step)

        # Confusion matrix
        summary_cm = utils.summary_confusion_matrix(self.config, gs_labels_gt, gs_labels_pred, self.config.labels, tensor_name='val/confusion_matrix')
        # summary_roc = utils.summary_roc(self.config, gs_labels_gt, gs_labels_pred, self.config.labels, tensor_name='val/roc')
        self.summary_writer.add_summary(summary_cm, global_step)
        # self.summary_writer.add_summary(summary_roc, global_step)


    def evaluate(self):
        epoch = 0
        self.run_epoch_eval(mode='eval', epoch=epoch)


    def predict(self, dataset_split_name='ds_test'):

        if dataset_split_name == 'ds_train':
            image_paths = utils_image.get_images_path_list_from_dir(self.config.tfrecords_path_train, img_format='jpg')
        elif dataset_split_name == 'ds_val':
            images_path = utils_image.get_images_path_list_from_dir(self.config.tfrecords_path_val, img_format='jpg')
        elif dataset_split_name == 'ds_test':
            images_path = utils_image.get_images_path_list_from_dir(self.config.tfrecords_path_test, img_format='jpg')
        else:
            logging.error('Unknown dataset_split_name {}', dataset_split_name)
            exit(1)


        ## Get image-label mapping
        image_label_dict = {}
        dataset_labels_file_path = 'datasets/densenet/ISIC2018_Task3_Training_GroundTruth.csv'
        # dataset_labels_file_path = 'ISIC2018_Task3_Training_GroundTruth.csv'
        with open(dataset_labels_file_path) as csvfile:
            read_csv = csv.reader(csvfile, delimiter=',')
            for index, row in enumerate(read_csv):
                ## Skip header
                if index == 0:
                    continue
                label_one_hot_encoding = [int(round(float(row[i+1]), 0)) for i in range(7)]
                image_label_dict[row[0]] = np.argmax(label_one_hot_encoding)


        
        ## Sample n images
        # random.shuffle(images_path)
        images_path = images_path[0:self.config.predict_num_images]


        ## Get labels_gt
        labels_gt = []
        for image_path in images_path:
            # TODO: Image name should have no dot
            # image_name = os.path.basename(image_path).split('.', 1)[0]
            image_name = os.path.basename(image_path).rsplit('.', 1)[0]
            labels_gt.append(image_label_dict[image_name])


        images_path = images_path[0:self.config.predict_num_images]
        images = []
        for image_path in images_path:

            ## Load image
            image = Image.open(image_path)

            ## Resize and center crop image. size: (width, height)
            image = ImageOps.fit(image, (self.config.tfr_image_width, self.config.tfr_image_height), Image.LANCZOS, 0, (0.5, 0.5))

            ## Preprocess images
            image = np.float32(np.array(image))
            image = self.data.preprocess_data(image)

            images.append(image)

        features = np.array(images)
        logging.debug('model_name {}'.format(self.config.model_name))
        logging.debug('features {}'.format(features.shape))


        ## Predict in batches
        num_images = len(images_path)
        batch_size = self.config.batch_size_pred
        iters = int(num_images/batch_size)
        logging.debug('num_images {}'.format(num_images))
        logging.debug('batch_size {}'.format(batch_size))
        labels_pred_cls = []
        labels_pred_prob = []

        idx_start = 0
        idx_end = 0
        for iter_no in range(iters):
            idx_start = iter_no * batch_size
            idx_end = idx_start + batch_size
            logging.debug('idx:[{}-{}]'.format(idx_start, idx_end))

            labels_pred_prob_batch, labels_pred_cls_batch = self.sess.run([
                    self.model.labels_pred_prob,
                    self.model.labels_pred_cls, 
                    ],
                feed_dict={
                    self.model.features: features[idx_start:idx_end], 
                    }
                )

            logging.debug('labels_gt             {}'.format(np.array(labels_gt[idx_start: idx_end])))
            logging.debug('labels_pred_cls_batch {}'.format(labels_pred_cls_batch))
            # logging.debug('labels_pred_prob_batch {}'.format(labels_pred_prob_batch))

            labels_pred_cls = labels_pred_cls + labels_pred_cls_batch.tolist()
            labels_pred_prob = labels_pred_prob + labels_pred_prob_batch.tolist()

        ## For images < batch_size and For images which do not fit the last batch
        idx_start = iters * batch_size
        idx_end = idx_start + (num_images % batch_size)
        logging.debug('idx:[{}-{}]'.format(idx_start, idx_end))
        if(num_images % batch_size):
            labels_pred_prob_batch, labels_pred_cls_batch = self.sess.run([
                    self.model.labels_pred_prob,
                    self.model.labels_pred_cls, 
                    ],
                feed_dict={
                    self.model.features: features[idx_start:idx_end], 
                    }
                )
            logging.debug('labels_gt             {}'.format(labels_gt[idx_start: idx_end]))
            logging.debug('labels_pred_cls_batch {}'.format(labels_pred_cls_batch))
            # logging.debug('labels_pred_prob_batch {}'.format(labels_pred_prob_batch))

            labels_pred_cls = labels_pred_cls + labels_pred_cls_batch.tolist()
            labels_pred_prob = labels_pred_prob + labels_pred_prob_batch.tolist()


        for label_gt, label_pred_cls in zip(labels_gt, labels_pred_cls):
            print('GT, PRED: [{}, {}]'.format(label_gt, label_pred_cls))


        ### ANALYAIS ###

        ## Plot ROC curve
        utils.gen_roc_curve(self.config, labels_gt, labels_pred_prob, dataset_split_name)

        ## Plot PR Curve
        utils.gen_precision_recall_curve(self.config, labels_pred_cls, labels_pred_prob, dataset_split_name)

        ## Confusion matrix
        # confusion = tf.confusion_matrix(labels=labels_gt, predictions=labels_pred_cls, num_classes=self.config.num_classes)
        # logging.debug('Row(GT), Col(Pred)')
        # with tf.Session() as sess:
        #     print(sess.run(confusion))


        ## Plot and save confusion matrix
        utils.get_confusion_matrix(self.config, labels_gt, labels_pred_cls, dataset_split_name)

        ## Print PR and F1
        utils.summary_pr_fscore(self.config, labels_gt, labels_pred_cls, self.config.labels)

        ## Plot Metrics
        utils.get_metrics(self.config, labels_gt, labels_pred_cls, dataset_split_name)



