##########################################################################################
# Main
##########################################################################################
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.python import debug as tf_debug

import os
import shutil
import random
import numpy as np
os.sys.path.append('./')

from data_handler.data_generator_knifey_spoony import DataGeneratorKnifeySpoony
from data_handler.tfrecords_knifey_spoony import TFRecordsKnifeySpoony
from models.model_knifey_spoony import ModelKnifeySpoony
from trainers.trainer_knifey_spoony import TrainerKnifeySpoony
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
        exit(0)

    if not os.path.exists(config.dataset_path_train):
        print(config.dataset_path_train)
        print('ERROR: Dataset not found')
        exit(0)


    ## Set log levels
    tf.logging.set_verbosity(tf.logging.INFO)


    ## Set seed values to reproduce results
    # random.seed(7)
    # np.random.seed(7)


    ## Create output dirs
    # utils.remove_dirs([os.path.join('output', config.exp_name)])
    utils.create_dirs([config.summary_dir, config.checkpoint_dir, config.tfrecords_path_train,
                      config.tfrecords_path_test])

    ## GPU
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth=True


    ## Create tensorflow session
    with tf.Session(config=sess_config) as sess:
        ## Debug
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)


        ## Create tensorboard logger
        logger = TFLogger(sess, config)


        ## Create TF Records
        TFRecordsKnifeySpoony(config)


        ## Create data generator using TF Records
        data = DataGeneratorKnifeySpoony(config)


        ## Create model
        model = ModelKnifeySpoony(config)


        ## Create Trainer
        trainer = TrainerKnifeySpoony(sess, model, data, config, logger)

        ## Train model
        trainer.train()
        trainer.evaluate()
        trainer.predict()


if __name__ == '__main__':
    main()
