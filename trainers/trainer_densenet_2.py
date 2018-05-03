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


from data_handler.data_generator_densenet import DataGeneratorDensenet
from data_handler.tfrecords_densenet import TFRecordsDensenet
from models.model_densenet import ModelDensenet
from trainers.trainer_densenet import TrainerDensenet
from utils.config import process_config
from utils.tf_logger import TFLogger
import utils.utils as utils
import utils.utils_image as utils_image



class TrainerDensenet(BaseTrain):
    def __init__(self, sess, model, data, config, logger, context, filenames):
        super(TrainerDensenet, self).__init__(sess, model, data, config,logger)
        self.context = context
        self.filenames = filenames


    def run_epochs(self, iters_done, summary_writer, epoch, model2, mode='TRAIN'):

        next_batch = data.input_fn(filenames=self.filenames, 
                                    train=True, 
                                    batch_size=self.config.batch_size, 
                                    buffer_size=self.config.data_gen_buffer_size),

        data_batch = sess.run(next_batch)
        # images_batch, labels_batch = sess.run(next_batch)
        images_batch = data_batch[0][0]['input_1']
        labels_batch = data_batch[0][1]

        # one_hot_labels = np.int32(keras.utils.to_categorical(labels_batch, num_classes=self.config.num_classes))
        one_hot_labels = labels_batch

        # Train the model, iterating on the data in batches of 32 samples
        model2.fit(images_batch, one_hot_labels, epochs=10, batch_size=self.config.batch_size)



if __name__ == '__main__':

    ## Configuration
    try:
        args = utils.get_args()
        config = process_config(args)
    except:
        print("missing or invalid arguments")
        exit(0)

    ## Logger
    utils.logger_init(config, logging.DEBUG) 

    # debug_mode = False
    # if len(sys.argv) > 1:
    #     if sys.argv[1] == '--debug':
    #         debug_mode = True
    #         logging.debug('Argument: {}'.format(sys.argv[1]))

    ## GPU
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True


    ## Save context
    context = {}
    context['epoch'] = 0
    context['step'] = 0
    context['iterations'] = 0
    context['iters_done'] = 0
    context['iters'] = 0
    context['cost'] = 0
    context['feat'] = 0
    context['timestamp_start'] = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
    context['timestamp_end'] = ''


    # Initialize Logger
    utils.logger_init(config, logging.DEBUG) 
    tf.logging.set_verbosity(tf.logging.DEBUG)


    ## Register signal handler
    utils.signal_handler(config)

    ## Set seed values to reproduce results
    random.seed(config.seed)
    np.random.seed(config.seed)
    tf.set_random_seed(config.seed)


    ## Create output dirs
    # utils.remove_dirs([os.path.join(config.output_path, config.exp_name)])
    utils.create_dirs([config.summary_dir, config.checkpoint_dir, config.tfrecords_path_train,
                      config.tfrecords_path_val, config.tfrecords_path_test])

    ## Save code
    utils.save_code(config)
    
    with tf.Session(config=sess_config) as sess:

        ## Initialze/Load variables
        # latest_checkpoint = tf.train.latest_checkpoint('output')
        # if config.pre_trained_model_path:
        #     models[0].saver.restore(sess, config.pre_trained_model_path)
        #     logging.debug('Restored pre_trained_model_path {}'.format(config.pre_trained_model_path))
        # elif latest_checkpoint:
        #     models[0].saver.restore(sess, latest_checkpoint)
        #     logging.debug('Restored latest_checkpoint {}'.format(config.pre_trained_model_path))
        # else:
        #     sess.run(tf.global_variables_initializer())

        sess.run(tf.global_variables_initializer())

        ## Create TF Records
        if (config.mode == 'tfr'):
            TFRecordsDensenet(config)
            exit(0)


        ## Create data generator using TF Records
        data = DataGeneratorDensenet(config)
        filenames_regex = os.path.join(config.tfrecords_path_train, '*.tfr')
        filenames = glob.glob(filenames_regex)
        logging.debug('filenames {}'.format(filenames))

        ## Create model
        model = ModelDensenet(config)
        model2 = model.build_model()


        ## Create Trainer
        logger = ''
        trainer = TrainerDensenet(sess, model, data, config, logger, context, filenames)

        ## TRAINING
        # trainer.train()
        iters_done = 0
        summary_writer = ''
        epoch = 0
        trainer.run_epochs(iters_done, summary_writer, epoch, model2, mode='TRAIN')



