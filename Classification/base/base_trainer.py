##########################################################################################
# Base class for Training
##########################################################################################
import tensorflow as tf

class BaseTrain:
    def __init__(self, sess, model, data, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.data = data

    def train(self):
        raise NotImplementedError

