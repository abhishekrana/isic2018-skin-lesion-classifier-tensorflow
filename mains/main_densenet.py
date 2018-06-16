##########################################################################################
# Main
##########################################################################################
import os
os.sys.path.append('./')
import shutil
import time
import random
import numpy as np
import logging
logging.getLogger('parso.python.diff').setLevel('INFO')
import datetime
from bunch import Bunch

import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.python import debug as tf_debug
from tensorflow.python.keras._impl.keras import backend as K

from data_handler.tfrecords_densenet import TFRecordsDensenet
from data_handler.data_generator_densenet import DataGeneratorDensenet
from models.model_densenet import ModelDensenet
from trainers.trainer_densenet import TrainerDensenet
from trainers.trainer_densenet_2 import TrainerDensenet_2
from utils.config import process_config
import utils.utils as utils
import utils.utils_image as utils_image
# from utils.tf_logger import TFLogger


def main():

    # Capture the config path from the run arguments then process the json configuration file
    try:
        args = utils.get_args()
        config = process_config(args)
    except:
        print("missing or invalid arguments")
        args={}
        args['config_file'] = 'configs/config_densenet.json'
        args['mode'] = 'predict'
        args = Bunch(args)
        config = process_config(args)


    if not os.path.exists(config.dataset_path_train):
        print(config.dataset_path_train)
        print('ERROR: Dataset not found')
        exit(1)

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


    ## Create tensorboard logger
    # logger = TFLogger(sess, config)


    # Config Keras
    K.set_image_data_format('channels_last')
    logging.debug('K.image_data_format {}'.format(K.image_data_format()))


    ## Configure GPU
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth=True
    gpu_train_fraction = 0.7
    if config.mode == 'train':
        sess_config.gpu_options.per_process_gpu_memory_fraction = gpu_train_fraction
    else:
        sess_config.gpu_options.per_process_gpu_memory_fraction = 1 - gpu_train_fraction
    tf.keras.backend.set_session(tf.Session(config=sess_config))


    ## TODO: _aSk Check why adding keras needed?
    # config.checkpoint_dir = os.path.join(config.checkpoint_dir, 'keras')

    timestamp_start = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
    logging.debug('timestamp_start {}'.format(timestamp_start))
    logging.debug('mode {}'.format(config.mode))

    ## Print updated configuration
    logging.debug('\nConfiguration: \n{}'.format(config))
    
    # sess=''
    logger=''
    ## Create TF Records
    if (config.mode == 'tfr'):
        TFRecordsDensenet(config)
        exit(0)


    ## Create data generator using TF Records
    data = DataGeneratorDensenet(config)


    ## Get and set class weights in config
    utils.set_config_class_weights(config, data)



    ## Create model
    model = ModelDensenet(config)
    model.build_model(config.mode)

    
    with tf.Session(config=sess_config) as sess:

        if config.debug == '1':
            # tf.train.start_queue_runners(sess=sess)
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess, thread_name_filter="MainThread$")

        ## Initialze/Load variables
        latest_checkpoint = tf.train.latest_checkpoint(config.checkpoint_dir)
        if config.pre_trained_model:
            model.saver.restore(sess, config.pre_trained_model)
            logging.debug('Restored pre_trained_model_path {}'.format(config.pre_trained_model))
        elif latest_checkpoint:
            model.saver.restore(sess, latest_checkpoint)
            logging.debug('Restored latest_checkpoint {}'.format(latest_checkpoint))
        else:
            sess.run(tf.global_variables_initializer())

        sess.run(tf.local_variables_initializer())

        # file_pattern=os.path.join(config.tfrecords_path_train, '*.tfr'),
        # filenames_op = tf.data.Dataset.list_files(file_pattern)
        # # filenames = sess.run(filenames_op)
        # logging.debug('filenames {}'.format(y))

        ## Create Trainer
        # trainer = TrainerDensenet(sess, model, data, config, logger)
        trainer = TrainerDensenet_2(sess, model, data, config, logger)
        

        ## TRAINING
        if (config.mode == 'train'):
            trainer.train()
            # trainer.train_and_eval()
            # trainer.train_tf()


        ## EVALUATION
        elif (config.mode == 'eval'):
            last_checkpoint = None
            while(True):

                latest_checkpoint = tf.train.latest_checkpoint(config.checkpoint_dir)

                if latest_checkpoint is None:
                    logging.debug('No checkpoint exists {}'.format(last_checkpoint))
                    time.sleep(5)
                    continue

                if last_checkpoint == latest_checkpoint:
                    logging.debug('Sleeping latest_checkpoint {}'.format(latest_checkpoint))
                    time.sleep(5)
                    continue

                last_checkpoint = latest_checkpoint
                model.saver.restore(sess, latest_checkpoint)

                trainer.evaluate()


        ## PREDICTION
        elif (config.mode == 'predict'):
            # trainer.predict(dataset_split_name='ds_train')
            trainer.predict(dataset_split_name='ds_val')
            trainer.predict(dataset_split_name='ds_test')


        else:
            logging.debug("ERROR: Unknown mode")
            exit(1)


if __name__ == '__main__':
    main()

"""
GPU: 8GB

"batch_size": 16,
"batch_size_eval": 2,

gpu_train_fraction = 0.7
"""


