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
os.sys.path.append('Classification/')


from data_handler.data_generator_cifar10 import DataGeneratorCifar10
from data_handler.tfrecords_cifar10 import TFRecordsCifar10
from models.model_cifar10 import ModelCifar10
from trainers.trainer_cifar10 import TrainerCifar10
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
        config_file = 'configs/config_cifar10.json'
        config = process_config(config_file)
        exit(0)

    if not os.path.exists(config.dataset_path_train):
        print(config.dataset_path_train)
        print('ERROR: Dataset not found')
        exit(0)


    ## Set log levels
    tf.logging.set_verbosity(tf.logging.DEBUG)


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
    # with tf.Session(config=sess_config) as sess:
        ## Debug
        # Part of the convenience provided by these APIs is that they manage Sessions internally.
        # This makes the LocalCLIDebugWrapperSession described in the preceding sections inapplicable.
        # Fortunately, you can still debug them by using special hooks provided by tfdbg
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        ## Create tensorboard logger
        # logger = TFLogger(sess, config)


    sess=''
    logger=''
    ## Create TF Records
    # TFRecordsCifar10(config)


    ## Create data generator using TF Records
    data = DataGeneratorCifar10(config)


    ## Create model
    model = ModelCifar10(config)


    ## Create Trainer
    trainer = TrainerCifar10(sess, model, data, config, logger)

    ## Train model
    # trainer.train()
    trainer.evaluate()
    trainer.predict()


if __name__ == '__main__':
    main()
