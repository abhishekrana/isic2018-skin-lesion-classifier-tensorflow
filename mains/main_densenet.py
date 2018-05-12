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
import datetime
from bunch import Bunch

import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.python import debug as tf_debug

from data_handler.tfrecords_densenet import TFRecordsDensenet
from data_handler.data_generator_densenet import DataGeneratorDensenet
from models.model_densenet import ModelDensenet
from trainers.trainer_densenet import TrainerDensenet
from utils.config import process_config
import utils.utils as utils
from utils.tf_logger import TFLogger


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


    sess=''
    logger=''
    ## Create TF Records
    if (config.mode == 'tfr'):
        TFRecordsDensenet(config)
        exit(0)


    ## Create data generator using TF Records
    data = DataGeneratorDensenet(config)


    ## Get and set class weights in config
    utils.set_config_class_weight(config, data)


    ## Create model
    model = ModelDensenet(config)


    ## Create Trainer
    trainer = TrainerDensenet(sess, model, data, config, logger)


    ## TODO: _aSk Check why adding keras needed?
    config.checkpoint_dir = os.path.join(config.checkpoint_dir, 'keras')

    timestamp_start = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
    logging.debug('timestamp_start {}'.format(timestamp_start))
    logging.debug('mode {}'.format(config.mode))
    logging.debug('\nConfiguration: \n{}'.format(config))


    ## TRAINING
    if (config.mode == 'train'):
        # trainer.train()
        trainer.train_and_eval()


    ## EVALUATION
    elif (config.mode == 'eval'):
        last_checkpoint = None
        while(True):

            latest_checkpoint = tf.train.latest_checkpoint(config.checkpoint_dir)

            if latest_checkpoint is None:
                logging.debug('No checkpoint does not exist {}'.format(last_checkpoint))
                time.sleep(5)
                continue

            if last_checkpoint == latest_checkpoint:
                logging.debug('Sleeping latest_checkpoint {}'.format(latest_checkpoint))
                time.sleep(5)
                continue

            last_checkpoint = latest_checkpoint

            trainer.evaluate()


    ## PREDICTION
    elif (config.mode == 'predict'):
        trainer.predict()


    else:
        logging.debug("ERROR: Unknown mode")
        exit(1)


if __name__ == '__main__':
    main()

