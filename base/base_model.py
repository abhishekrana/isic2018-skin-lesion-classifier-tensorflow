##########################################################################################
# Base class for creating Model
##########################################################################################
import tensorflow as tf

class BaseModel:
    def __init__(self, config):
        self.config = config
        self.build_model()

    def build_model(self):
        raise NotImplementedError
