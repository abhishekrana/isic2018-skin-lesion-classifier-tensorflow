##########################################################################################
# Model

# Estimators
# pre-made Estimators create and manage Graph and Session objects for you
#
# https://www.dlology.com/blog/an-easy-guide-to-build-new-tensorflow-datasets-and-estimator-with-epeat_count=5,
##########################################################################################
from base.base_model import BaseModel
import tensorflow as tf
import os
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras import models
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D

from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Input, AveragePooling2D, Activation
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras.optimizers import Adam, SGD

# from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.applications.inception_v3 import InceptionV3

from models.model_densenet_121 import Densenet121
from models.model_densenet_169 import Densenet169
from models.model_densenet_161 import Densenet161
from models.model_densenet_custom_layers import Scale


class ModelDensenet(BaseModel):
    def __init__(self, config):
        super(ModelDensenet, self).__init__(config)


    def build_model(self):
        """
        # config = tf.estimator.RunConfig(
        #         model_dir=None,                                                                   # None: will use a default value set by the Estimator.
        #         tf_random_seed=None,
        #         save_summary_steps=100,
        #         save_checkpoints_steps=_USE_DEFAULT,
        #         save_checkpoints_secs=_USE_DEFAULT,
        #         session_config=None,
        #         keep_checkpoint_max=5,
        #         keep_checkpoint_every_n_hours=10000,
        #         log_step_count_steps=100
        #     )
        """
        params = {"learning_rate": self.config.learning_rate}


        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth=True
        est_config =  tf.estimator.RunConfig(
                                            session_config=session_config,
                                            save_checkpoints_steps=self.config.train_save_checkpoints_steps,
                                            save_summary_steps=self.config.train_save_summary_steps)

        ## Keras Model
        model_dir = os.path.join(os.getcwd(), self.config.checkpoint_dir, 'keras') 

        model = self.model_vgg16()

        model.summary()

        # model.compile(loss='binary_crossentropy',
        # model.compile(loss='categorical_crossentropy',
        model.compile(loss='sparse_categorical_crossentropy',
                optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
                metrics=['acc'])

        self.model_estimator = tf.keras.estimator.model_to_estimator(keras_model=model,
                                                                     config=est_config,
                                                                     custom_objects=params,
                                                                     model_dir=model_dir)

    def model_vgg16(self):
        # https://github.com/Tony607/Keras_catVSdog_tf_estimator/blob/master/keras_estimator_vgg16-cat_vs_dog.ipynb
        # https://github.com/Tony607/Keras_catVSdog_tf_estimator/blob/master/keras_estimator_vgg16-cat_vs_dog-TFRecord.ipynb
        # https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.3-using-a-pretrained-convnet.ipynb

        # img_size = (150, 150, 3)
        img_size = (self.config.tfr_image_height, self.config.tfr_image_width, self.config.tfr_image_channels)
        conv_base = VGG16(weights='imagenet',
                        include_top=False,
                        input_shape=img_size)

        model = models.Sequential()
        model.add(conv_base)
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(self.config.num_classes, activation='sigmoid'))

        conv_base.trainable = False

        return model



