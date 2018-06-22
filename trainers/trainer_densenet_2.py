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
from heapq import nlargest
import timeit

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


    def update_prediction(self, predict_dict, features, labels_gt):

        # total_predict_index = self.config.num_images_per_fold/self.config.batch_size
        # predict_batch = self.config.batch_patch

        # TODO: Handle last batch
        iters = int(self.config.num_images_per_fold/self.config.batch_size)

        # for iter_no in range(total_predict_index) :
        for iter_no in range(iters):
            # result = dict()

            idx_start = iter_no * self.config.batch_size
            idx_end = idx_start + self.config.batch_size

            if (idx_end <= len(labels_gt)):
                features_batch = features[idx_start: idx_end]
                labels_gt_batch = labels_gt[idx_start: idx_end]
            else:
                features_batch = features[idx_start: len(features_batch)]
                labels_gt_batch = labels_gt[idx_start: len(features_batch)]
            # features_batch = features[predict_batch*p_idx : predict_batch*(p_idx+1)]
            # labels_gt_batch = labels_gt[predict_batch*p_idx : predict_batch*(p_idx+1)]

            # TODO: Handle case for last batch

            # start_time2 = timeit.default_timer()
            cross_entropy = self.sess.run([
                        self.model.cross_entropy
                        ],
                    feed_dict={
                        self.model.features: features_batch, 
                        self.model.labels: labels_gt_batch
                        }
                    )
            # print('start2: {}', timeit.default_timer() - start_time2)

            for idx in range(idx_start, idx_end):
                predict_dict.update({idx: cross_entropy[0][(idx%self.config.batch_size)]})


            # result = {idx_start+i: loss[i] for i in range(self.config.batch_size)}

            # temp_x = self.sess.run(self.x_dict, feed_dict=predictor_feed_dict)
            # if iter_no != 0 :
            #     for k,v in temp_x.items():
            #         new_k = k + predict_batch * iter_no
            #         result[new_k] = v
            # else :
            #     for k,v in temp_x.items():
            #         result[k] = v

            # predict_dict.update(result)
            # index = nlargest(self.batch_size, predict_dict, key=predict_dict.get)
            # predict_dict = {s_idx: predict_dict[s_idx] for s_idx in index}

        # g_r_index = list(predict_dict.keys())
        # features_batch = features[g_r_index]
        # labels_gt_batch = labels_gt[g_r_index]



    def run_epoch_train_adaptive(self, mode, epoch):

        assert (mode=='train')

        # TODO: Will loose last fold here
        k_folds = int(self.config.debug_train_images_count/self.config.num_images_per_fold)

        for k_fold in range(k_folds):
            # logging.debug('k_fold {}'.format(k_fold))

            data_batch = self.data.input_fn(
                                        file_pattern=os.path.join(self.config.tfrecords_path_train, '*.tfr'),
                                        mode=mode,
                                        batch_size=self.config.num_images_per_fold,
                                        buffer_size=self.config.data_gen_buffer_size
                                        )
            features_dict, labels_gt = self.sess.run(data_batch)
            features = features_dict[self.config.model_name + '_input']

            predict_dict = dict()
            # self.update_prediction(predict_dict, features, labels_gt)

            num_steps = int(len(features)/self.config.batch_size)

            idx_start = 0
            idx_end = 0
            num_nlargest_per_fold = 256
            nlargest_index = []
            for step in range(num_steps):
                idx_start = step * self.config.batch_size
                idx_end = idx_start + self.config.batch_size
                # logging.debug('idx:[{}-{}]'.format(idx_start, idx_end))

                # features_batch = features[idx_start: idx_end]
                # labels_gt_batch = labels_gt[idx_start: idx_end]

                # nlargest_index = nlargest(self.config.batch_size, predict_dict, key=predict_dict.get)
                # for idx in nlargest_index:
                #     logging.debug('idx Loss : {:4} : {:8.6}'.format(idx, predict_dict[idx]))

                # nlargest_index = nlargest(num_nlargest_per_fold, predict_dict, key=predict_dict.get)



                idx_start_nlargest = (step * self.config.batch_size)%num_nlargest_per_fold
                idx_end_nlargest = idx_start_nlargest + self.config.batch_size
                # logging.debug('idx_nlargest {}-{}'.format(idx_start_nlargest, idx_end_nlargest))

                if int(step*self.config.batch_size%num_nlargest_per_fold) == 0:
                    logging.debug('update_prediction')
                    self.update_prediction(predict_dict, features, labels_gt)
                    nlargest_index = nlargest(num_nlargest_per_fold, predict_dict, key=predict_dict.get)
                    # logging.debug('nlargest_index {}'.format(nlargest_index))
                    # for idx in nlargest_index:
                    #     logging.debug('idx Loss : {:4} : {:8.6}'.format(idx, predict_dict[idx]))

                nlargest_index_batch = nlargest_index[idx_start_nlargest:idx_end_nlargest]
                features_batch = features[nlargest_index_batch]
                labels_gt_batch = labels_gt[nlargest_index_batch]

                _, loss, metrics, summary = self.sess.run([
                                        self.model.train_op,
                                        self.model.loss,
                                        self.model.metrics,
                                        self.model.summary_op,
                                        # self.model.summary_pr_op,
                                        # self.model.tf_print_op
                                        ],
                                    feed_dict={
                                        self.model.features: features_batch,
                                        self.model.labels: labels_gt_batch
                                        }
                                    )

                # start_time = timeit.default_timer()
                # self.update_prediction(predict_dict, features, labels_gt)
                # print('start1: {}', timeit.default_timer() - start_time)

                # global_step refer to the number of batches seen by the graph. When it is passed in the
                # optimizer.minimize() argument list, the variable is increased by one
                global_step = self.sess.run(tf.train.get_global_step())

                # logging.debug('Epoch:{}, global_step:{}, step:{}, loss:{}, accuracy:{}, metrics:{}'.format(epoch, global_step, step, loss, metrics['accuracy'], metrics))
                # logging.debug('Epoch:{}, k_fold:{}, global_step:{}, step:{}, loss:{}, accuracy:{}, metrics:{}'.format(epoch, k_fold, global_step, step, loss, metrics['accuracy'], metrics))
                # logging.debug('Epoch:{}, global_step:{}, k_fold:{}/{}, step:{}/{}, idx:{}-{} loss:{}, accuracy:{}'.format(epoch, global_step, k_fold, k_folds, step, num_steps, idx_start, idx_end, loss, metrics['accuracy']))
                logging.debug('Epoch:{}, global_step:{}, k_fold:{}/{}, step:{}/{}, idx:{}-{}, idx_nlargest:{}-{}, loss:{:.4}, accuracy:{:.4}'.format(epoch, global_step, k_fold, k_folds, step, num_steps, idx_start, idx_end, idx_start_nlargest, idx_end_nlargest, loss, metrics['accuracy'][0]))

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




            ## TODO: Handle this case
            # # For images < batch_size and images which do not fit the last batch
            # idx_start = iters * batch_size
            # idx_end = idx_start + (num_images % batch_size)
            # print('\nidx:[{}-{}]'.format(idx_start, idx_end))
            # if(num_images % batch_size):
            #     output_path_mod = os.path.join(output_path, 'record_' + str(iters) + '.tfr')
            #     self.create_tfrecord(image_paths, labels, idx_start, idx_end, output_path_mod)


    def train(self):
        epoch = 0
        for epoch in range(self.config.num_epochs):
            # self.run_epoch_train(mode='train', epoch=epoch)
            self.run_epoch_train_adaptive(mode='train', epoch=epoch)


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



