import tensorflow as tf

import os
import shutil
import random
import numpy as np
os.sys.path.append('./')
# os.sys.path.append('/home/abhishek/Abhishek/git/MLMI_ISIC2018/Segmentation')

from data_loader.data_generator import DataGenerator
from data_loader.data_generator_mnist import DataGeneratorMnist
from data_loader.tfrecords_generator import TFRecordsGenerator
from models.example_model import ExampleModel
from models.mnist_model import MnistModel
from trainers.example_trainer import ExampleTrainer
from trainers.mnist_trainer import MnistTrainer
from utils.config import process_config
from utils.tf_logger import TFLogger
import utils.utils as utils


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = utils.get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)


    if not os.path.exists(config.dataset_path_train):
        print(config.dataset_path_train)
        print('ERROR: Dataset not found')
        # TODO
        # exit(0)

    tf.logging.set_verbosity(tf.logging.INFO)

    ## Set seed values to reproduce results
    random.seed(7)
    np.random.seed(7)

    ## GPU
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth=True

    ## Create output dirs
    utils.remove_dirs([os.path.join('output', config.exp_name)])
    utils.create_dirs([config.summary_dir, config.checkpoint_dir, config.tfrecords_path_train,
                      config.tfrecords_path_test])

    ## Create tensorflow session
    sess = tf.Session()

    ## Create model
    #model = ExampleModel(config)
    model = MnistModel(config)

    ## Load model if exists
    #model.load(sess)

    ## Create data generator
    # TFRecordsGenerator(config)
    # data = DataGenerator(config)
    data = DataGeneratorMnist(config)

    ## Create tensorboard logger
    logger = TFLogger(sess, config)

    ## Create trainer and pass all the previous components to it
    # trainer = ExampleTrainer(sess, model, data, config, logger)
    trainer = MnistTrainer(sess, model, data, config, logger)

    ## Train model
    # trainer.train()
    trainer.train_estimator()


if __name__ == '__main__':
    main()
