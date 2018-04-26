##########################################################################################
# Main
##########################################################################################
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.python import debug as tf_debug

import os
import shutil
import time
import random
import numpy as np
os.sys.path.append('./')
os.sys.path.append('Classification/')


from data_handler.data_generator_densenet import DataGeneratorDensenet
from data_handler.tfrecords_densenet import TFRecordsDensenet
from models.model_densenet import ModelDensenet
from trainers.trainer_densenet import TrainerDensenet
from utils.config import process_config
from utils.tf_logger import TFLogger
import utils.utils as utils


def main():
    # Capture the config path from the run arguments then process the json configuration file
    try:
        args = utils.get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        config_file = 'configs/config_densenet.json'
        config = process_config(config_file)
        # exit(0)

    if not os.path.exists(config.dataset_path_train):
        print(config.dataset_path_train)
        print('ERROR: Dataset not found')


    ## Set log levels
    tf.logging.set_verbosity(tf.logging.DEBUG)


    ## Set seed values to reproduce results
    random.seed(config.seed)
    np.random.seed(config.seed)


    ## Create output dirs
    # utils.remove_dirs([os.path.join('output', config.exp_name)])
    utils.create_dirs([config.summary_dir, config.checkpoint_dir, config.tfrecords_path_train,
                      config.tfrecords_path_test])


    ## Create tensorboard logger
    # logger = TFLogger(sess, config)


    sess=''
    logger=''
    ## Create TF Records
    ## TODO: Regenerate records everytime `batch_size` is changed
    # TFRecordsDensenet(config)

    ## Create data generator using TF Records
    data = DataGeneratorDensenet(config)


    ## Create model
    model = ModelDensenet(config)


    ## Create Trainer
    trainer = TrainerDensenet(sess, model, data, config, logger)


    print('MODE: {}'.format(args.mode))
    # checkpoints_path = os.path.join('output', config.exp_name, 'checkpoints')
    checkpoints_path = os.path.join('output', config.exp_name, 'checkpoints', 'keras')

    ## EVALUATION
    if (args.mode == 'eval'):
        last_checkpoint = None
        while(True):

            latest_checkpoint = tf.train.latest_checkpoint(checkpoints_path)

            if latest_checkpoint is None:
                print('No checkpoint does not exist {}'.format(last_checkpoint))
                time.sleep(5)
                continue

            if last_checkpoint == latest_checkpoint:
                print('Sleeping latest_checkpoint {}'.format(latest_checkpoint))
                time.sleep(5)
                continue

            last_checkpoint = latest_checkpoint

            trainer.evaluate()

    ## PREDICTION
    elif (args.mode == 'predict'):
        trainer.predict()

    ## TRAINING
    else:
        trainer.train()


if __name__ == '__main__':
    main()
