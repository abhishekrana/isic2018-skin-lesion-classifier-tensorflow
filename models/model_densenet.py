##########################################################################################
# Model

# Estimators
# pre-made Estimators create and manage Graph and Session objects for you
#
# https://www.dlology.com/blog/an-easy-guide-to-build-new-tensorflow-datasets-and-estimator-with-keras-model/
##########################################################################################
from base.base_model import BaseModel
import tensorflow as tf
import os
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras import models
from tensorflow.python.keras import layers

from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Input, AveragePooling2D, Activation
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras.optimizers import Adam, SGD, RMSprop

from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201


class ModelDensenet(BaseModel):
    def __init__(self, config):
        super(ModelDensenet, self).__init__(config)


    def build_model(self):

        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth=True
        est_config =  tf.estimator.RunConfig(
                                            # model_dir=None,             # None: Use a default value set by the Estimator.
                                            # tf_random_seed=None,
                                            # save_checkpoints_secs=_USE_DEFAULT,
                                            # keep_checkpoint_max=5,
                                            # keep_checkpoint_every_n_hours=10000,
                                            # log_step_count_steps=100,
                                            session_config=session_config,
                                            save_checkpoints_steps=self.config.train_save_checkpoints_steps,
                                            save_summary_steps=self.config.train_save_summary_steps)

        ## Keras Model
        model = self.model_densenet121()
        model.summary()


        optimizer = SGD(lr=self.config.learning_rate, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5.0)
        # optimizer = RMSprop(lr=2e-5)

        # model.compile(loss='sparse_categorical_crossentropy',
        model.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                weighted_metrics=self.config.class_weight,
                metrics=['acc'])


        ## Create Estimator from Keras model
        params = {"learning_rate": self.config.learning_rate}
        model_dir = os.path.join(os.getcwd(), self.config.checkpoint_dir, 'keras')
        self.model_estimator = tf.keras.estimator.model_to_estimator(keras_model=model,
                                                                     config=est_config,
                                                                     custom_objects=params,
                                                                     model_dir=model_dir)
        return model


    def model_densenet121(self):
	
        img_size = (self.config.tfr_image_height, self.config.tfr_image_width, self.config.tfr_image_channels)

        # Densenet121
        conv_base = DenseNet121(weights='imagenet', include_top=False, input_shape=img_size)

        model = models.Sequential()
        model.add(conv_base)

        model.add(layers.Flatten())

        model.add(layers.Dense(512, activation='relu'))
        model.add(Dropout(0.25))

        model.add(layers.Dense(512, activation='relu'))
        model.add(Dropout(0.25))

        model.add(layers.Dense(128, activation='relu'))

        # model.add(layers.Dense(self.config.num_classes, activation='softmax', name='class_prob'))
        model.add(layers.Dense(self.config.num_classes, activation='softmax', name='dense_4'))

        conv_base.trainable = False

        return model

