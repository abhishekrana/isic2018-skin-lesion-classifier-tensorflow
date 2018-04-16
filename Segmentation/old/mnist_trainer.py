
from base.base_trainer import BaseTrain
from tqdm import tqdm
import tensorflow as tf
import numpy as np



class MnistTrainer(BaseTrain):
    def __init__(self, sess, model, data, config,logger):
        super(MnistTrainer, self).__init__(sess, model, data, config,logger)


    def train_estimator(self):

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": np.array(self.data.train.images)},
            y=np.array(self.data.train.cls),
            num_epochs=None,
            shuffle=True)


        self.model.model_estimator.train(input_fn=train_input_fn, steps=2000)


    def train_epoch(self):
        pass


    def train_step(self):
        pass

