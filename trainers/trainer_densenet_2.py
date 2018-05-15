##########################################################################################
# Training
##########################################################################################
import os
os.sys.path.append('./')
os.sys.path.append('../')


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
from utils.tf_logger import TFLogger
import utils.utils as utils
import utils.utils_image as utils_image



class TrainerDensenet_2(BaseTrain):

    def __init__(self, sess, model, data, config,logger):
        super(TrainerDensenet_2, self).__init__(sess, model, data, config,logger)

        # sess.run(tf.initialize_all_variables())
        self.summary_writer = tf.summary.FileWriter(
                                                os.path.join(self.config.summary_dir, self.config.mode),
                                                graph=self.sess.graph, flush_secs=30)
            

    # def run_epoch(self, is_train, num_steps, iters_done, summary_writer, epoch):
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

            features_dict, labels = self.sess.run(data_batch)
            features = features_dict[self.config.model_name + '_input']

            loss, _, metrics, summary = self.sess.run([
                        self.model.loss, 
                        self.model.train_op, 
                        self.model.metrics,
                        self.model.summary_op
                        ],
                    feed_dict={
                        self.model.features: features, 
                        self.model.labels: labels
                        }
                    )

            # global_step refer to the number of batches seen by the graph. When it is passed in the 
            # optimizer.minimize() argument list, the variable is increased by one
            global_step = self.sess.run(tf.train.get_global_step())

            logging.debug('Epoch:{}, global_step:{}, step:{}, loss:{}, accuracy:{}'.format(epoch, global_step, step, loss, metrics))

            ## Save checkpoints
            if (global_step%self.config.train_save_checkpoints_steps) == 0:
                self.model.saver.save(
                        self.sess, 
                        save_path=os.path.join(self.config.checkpoint_dir, 'model_{}.ckpt'.format(global_step))
                        )

            ## Save summary
            if (global_step%self.config.train_save_summary_steps) == 0:
                self.summary_writer.add_summary(summary, global_step)



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
        
        for step in range(num_steps):

            try:
                features_dict, labels = self.sess.run(data_batch)
            except Exception:
                logging.debug('Input data stream read completely {}'.format(Exception))
                global_step = self.sess.run(tf.train.get_global_step())
                self.summary_writer.add_summary(summary, global_step)
                # pass
                return

            features = features_dict[self.config.model_name + '_input']

            loss, metrics, summary = self.sess.run([
                        self.model.loss, 
                        self.model.metrics,
                        self.model.summary_op
                        ],
                    feed_dict={
                        self.model.features: features, 
                        self.model.labels: labels
                        }
                    )

            # global_step refer to the number of batches seen by the graph. When it is passed in the 
            # optimizer.minimize() argument list, the variable is increased by one
            global_step = self.sess.run(tf.train.get_global_step())

            logging.debug('Epoch:{}, global_step:{}, step:{}, loss:{}, accuracy:{}'.format(epoch, global_step, step, loss, metrics))

            ## Save summary
            if (global_step%self.config.train_save_summary_steps) == 0:
                self.summary_writer.add_summary(summary, global_step)


    def evaluate(self):
        epoch = 0
        self.run_epoch_eval(mode='eval', epoch=epoch)


    def predict(self):

        ## Get image-label mapping
        image_label_dict = {}
        dataset_labels_file_path = 'datasets/densenet/ISIC2018_Task3_Training_GroundTruth.csv'
        with open(dataset_labels_file_path) as csvfile:
            read_csv = csv.reader(csvfile, delimiter=',')
            for index, row in enumerate(read_csv):
                ## Skip header
                if index == 0:
                    continue
                label_one_hot_encoding = [int(round(float(row[i+1]), 0)) for i in range(7)]
                image_label_dict[row[0]] = np.argmax(label_one_hot_encoding)


        ## Get image paths
        image_paths = utils_image.get_images_path_list_from_dir(self.config.tfrecords_path_train, img_format='jpg')
        # image_paths = utils_image.get_images_path_list_from_dir(self.config.tfrecords_path_test, img_format='jpg')
        # image_paths = utils_image.get_images_path_list_from_dir(self.config.tfrecords_path_val, img_format='jpg')

        ## Sample n images
        random.shuffle(image_paths)
        image_paths = image_paths[0:self.config.predict_num_images]


        ## Get labels_gt
        labels_gt = []
        for image_path in image_paths:
            image_name = os.path.basename(image_path).rsplit('.', 1)[0]
            labels_gt.append(image_label_dict[image_name])


        image_paths = image_paths[0:self.config.predict_num_images]
        images = []
        for image_path in image_paths:

            ## Load image
            image = Image.open(image_path)

            ## Resize and center crop image. size: (width, height)
            image = ImageOps.fit(image, (self.config.tfr_image_width, self.config.tfr_image_height), Image.LANCZOS, 0, (0.5, 0.5))


            # img = cv2.imread(image_path)
            # img = cv2.resize(img, (224, 224))


            ## Preprocess images
            image = np.float32(np.array(image))
            image = self.data.preprocess_data(image)

            images.append(image)

        features = np.array(images)
        logging.debug('model_name {}'.format(self.config.model_name))
        logging.debug('features {}'.format(features.shape))

        labels_pred, labels_pred_cls = self.sess.run([
                    self.model.labels_pred, 
                    self.model.labels_pred_cls, 
                    ],
                feed_dict={
                    self.model.features: features, 
                    # self.model.labels: labels
                    }
                )

        logging.debug('labels_pred_cls {}'.format(labels_pred_cls))
        logging.debug('labels_pred {}'.format(labels_pred))

        # # TODO: Don't shuffle else labels will mismatch
        # x_key = self.config.model_name + '_input'
        # predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        #         x={x_key: images},
        #         y=None,
        #         batch_size=self.config.batch_size_pred,
        #         num_epochs=1,
        #         shuffle=False,
        #         queue_capacity=1000,
        #         # In order to have predicted and repeatable order of reading and enqueueing,
        #         # such as in prediction and evaluation mode, num_threads should be 1.
        #         num_threads=1)

        # with tf.Session() as sess:
        #     init_op = tf.global_variables_initializer()
        #     sess.run(init_op)

        #     # sess.run(self.model.model.output, feed_dict={self.model.model.input:images})
        #     # sess.run(self.model.model['class_prob'], feed_dict={self.model.model.input:images})
        #     # op = self.model.model.get_layer('class_prob')
        #     # op = self.model.model.get_layer('class_prob')
        #     # op = self.model.model.layers[0]
        #     op = self.model.model.get_layer('dense_1')
        #     out = sess.run(op, 
        #             feed_dict={self.model.model.input:images})
        #     logging.debug('out {}'.format(out))
        # exit(0)


        # checkpoint_path = None
        # if not self.config.predict_weights_path:
        #     checkpoint_path = self.config.predict_weights_path

        # NOTE: predictions is <generator object Estimator.predict> and hence (maybe) we can dereference it only once.
        # TODO: Check predict_keys
        # predictions = self.estimator.predict(
        #                                 input_fn=predict_input_fn,
        #                                 checkpoint_path=checkpoint_path,
        #                                 hooks=None,
        #                                 predict_keys=None,
        #                                 yield_single_examples=True
        #                                 )

        # class_prob = [p['class_prob'] for p in predictions]
        # # class_prob = [p['dense_4'] for p in predictions]
        # pred_labels = np.argmax(np.array(class_prob), axis=1)

        for label_gt, label_pred_cls in zip(labels_gt, labels_pred_cls):
            print('GT, PRED: [{}, {}]'.format(label_gt, label_pred_cls))


        ## Confusion matrix
        # https://stackoverflow.com/questions/41617463/tensorflow-confusion-matrix-in-tensorboard
        confusion = tf.confusion_matrix(labels=labels_gt, predictions=labels_pred_cls, num_classes=self.config.num_classes)
        logging.debug('Row(GT), Col(Pred)')
        with tf.Session() as sess:
            print(sess.run(confusion))

        # # Plot and save confusion matrix
        # utils.get_confusion_matrix(self.config, labels_gt, pred_labels)





# if __name__ == '__main__':

#     ## Configuration
#     try:
#         args = utils.get_args()
#         config = process_config(args)
#     except:
#         print("missing or invalid arguments")
#         exit(0)

#     ## Logger
#     utils.logger_init(config, logging.DEBUG) 

#     # debug_mode = False
#     # if len(sys.argv) > 1:
#     #     if sys.argv[1] == '--debug':
#     #         debug_mode = True
#     #         logging.debug('Argument: {}'.format(sys.argv[1]))

#     ## GPU
#     sess_config = tf.ConfigProto()
#     sess_config.gpu_options.allow_growth = True


#     ## Save context
#     context = {}
#     context['epoch'] = 0
#     context['step'] = 0
#     context['iterations'] = 0
#     context['iters_done'] = 0
#     context['iters'] = 0
#     context['cost'] = 0
#     context['feat'] = 0
#     context['timestamp_start'] = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
#     context['timestamp_end'] = ''


#     # Initialize Logger
#     utils.logger_init(config, logging.DEBUG) 
#     tf.logging.set_verbosity(tf.logging.DEBUG)


#     ## Register signal handler
#     utils.signal_handler(config)

#     ## Set seed values to reproduce results
#     random.seed(config.seed)
#     np.random.seed(config.seed)
#     tf.set_random_seed(config.seed)


#     ## Create output dirs
#     # utils.remove_dirs([os.path.join(config.output_path, config.exp_name)])
#     utils.create_dirs([config.summary_dir, config.checkpoint_dir, config.tfrecords_path_train,
#                       config.tfrecords_path_val, config.tfrecords_path_test])

#     ## Save code
#     utils.save_code(config)
    
#     with tf.Session(config=sess_config) as sess:

#         ## Initialze/Load variables
#         # latest_checkpoint = tf.train.latest_checkpoint('output')
#         # if config.pre_trained_model_path:
#         #     models[0].saver.restore(sess, config.pre_trained_model_path)
#         #     logging.debug('Restored pre_trained_model_path {}'.format(config.pre_trained_model_path))
#         # elif latest_checkpoint:
#         #     models[0].saver.restore(sess, latest_checkpoint)
#         #     logging.debug('Restored latest_checkpoint {}'.format(config.pre_trained_model_path))
#         # else:
#         #     sess.run(tf.global_variables_initializer())

#         sess.run(tf.global_variables_initializer())

#         ## Create TF Records
#         if (config.mode == 'tfr'):
#             TFRecordsDensenet(config)
#             exit(0)


#         ## Create data generator using TF Records
#         data = DataGeneratorDensenet(config)
#         filenames_regex = os.path.join(config.tfrecords_path_train, '*.tfr')
#         filenames = glob.glob(filenames_regex)
#         logging.debug('filenames {}'.format(filenames))

#         ## Create model
#         model = ModelDensenet(config)
#         model2 = model.build_model()


#         ## Create Trainer
#         logger = ''
#         trainer = TrainerDensenet(sess, model, data, config, logger, context, filenames)

#         ## TRAINING
#         # trainer.train()
#         iters_done = 0
#         summary_writer = ''
#         epoch = 0
#         trainer.run_epochs(iters_done, summary_writer, epoch, model2, mode='TRAIN')



