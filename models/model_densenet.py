##########################################################################################
# Model

# Estimators
# pre-made Estimators create and manage Graph and Session objects for you
#
# https://www.dlology.com/blog/an-easy-guide-to-build-new-tensorflow-datasets-and-estimator-with-keras-model/
##########################################################################################
from base.base_model import BaseModel

import os
import logging

import tensorflow as tf
from tensorflow.python.keras import models
from tensorflow.python.keras import layers
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Input, AveragePooling2D, Activation
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras.optimizers import Adam, SGD, RMSprop

from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.applications.xception import Xception
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.python.keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
import utils.w_categorical_crossentropy as wcce
from functools import partial
import pickle

class ModelDensenet(BaseModel):

    def __init__(self, config):
        super(ModelDensenet, self).__init__(config)

        
    def build_model(self, mode):
        """
        mode: train/eval/test
        """
        assert (mode=='train') or (mode=='eval') or (mode=='predict')

        logging.debug('build_model')
        global_step = tf.Variable(0, name='global_step',trainable=False)

        # Inputs
        self.features = tf.placeholder(tf.float32, shape=[None, self.config.tfr_image_height, self.config.tfr_image_width, self.config.tfr_image_channels], name='features_ph')
        self.labels = tf.placeholder(tf.float32, shape=[None, self.config.num_classes], name='labels_ph')


        ## Model
        self.logits = self.create_model()


        self.labels_pred = tf.nn.softmax(logits=self.logits)
        self.labels_pred_cls = tf.argmax(input=self.labels_pred, axis=1)

        if (mode=='train') or (mode=='eval'):
            with open("../utils/R-50.pkl", "rb") as f:
                u = pickle._Unpickler(f)
                u.encoding = 'latin1'
                weights = u.load()

            # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
            # w_cross_entropy = wcce.w_categorical_crossentropy(self.labels, self.labels_pred_cls, weights)

            ## Loss
            epsilon = tf.constant(value=1e-10)
            logits = self.logits + epsilon
            #label_flat = tf.reshape(self.labels, (-1, 1))
            #labels = tf.reshape(tf.one_hot(label_flat, depth=self.config.num_classes), (-1, self.config.num_classes))
            #labels = tf.argmax(labels)
            labels = self.labels
            softmax = tf.nn.softmax(logits)
            cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax + epsilon), self.config.class_weight_dict), reduction_indices=[1])
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
            tf.add_to_collection('losses', cross_entropy_mean)
            self.loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

            # self.loss = tf.reduce_mean(cross_entropy)

            ## Optimizer
            if self.config.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
            elif self.config.optimizer == 'rms_prop':
                optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config.learning_rate)
            elif self.config.optimizer == 'gradient_descent':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.learning_rate)
            else:
                logging.error('Unknown optimizer {}'.format(self.config.optimizer))
                exit(1)


            ## Train op
            self.train_op = optimizer.minimize(loss=self.loss, global_step=tf.train.get_global_step())

            ## Evaluation metrics
            self.metrics = \
            {
                # "accuracy": tf.metrics.accuracy(labels, self.labels_pred_cls)
                "accuracy": tf.metrics.accuracy(tf.argmax(self.labels, axis=1), self.labels_pred_cls)
            }


        ## Save Model
        # TODO: Set up the Saver after setting up the AdamOptimizer because ADAM has state (namely per-weight learning rates) that need to be restored as well.
        self.saver = tf.train.Saver(max_to_keep=self.config.train_keep_checkpoint_max)


        if mode=='predict':
            logging.debug('mode {}'.format(mode))
            return


        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy/value', self.metrics['accuracy'][0])
        # tf.summary.scalar('accuracy/update_op', self.metrics['accuracy'][1])

        self.summary_op = tf.summary.merge_all()


    # def create_model(self, features_dict, labels, mode, params, config):
    def create_model(self):
        """
        https://github.com/tensorflow/tensorflow/issues/14356

        """

        input_shape = (self.config.tfr_image_height, self.config.tfr_image_width, self.config.tfr_image_channels)

        ## VGG16
        if self.config.model_name == 'vgg16':
            base_model = VGG16(weights='imagenet', include_top=False, input_tensor=self.features, input_shape=input_shape)

        ## Xception
        elif self.config.model_name == 'xception':
            base_model = Xception(weights='imagenet', include_top=False, input_tensor=self.features, input_shape=input_shape)

        ## Resnet50
        elif self.config.model_name == 'resnet50':
            base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=self.features, input_shape=input_shape)
            logits = self.model_top_resnet50(base_model)

        ## InceptionResNetV2
        elif self.config.model_name == 'inception_resnet_v2':
            base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_tensor=self.features, input_shape=input_shape)

        ## Densenet121
        elif self.config.model_name == 'densenet121':
            base_model = DenseNet121(weights='imagenet', include_top=False, input_tensor=self.features, input_shape=input_shape)
            logits = self.model_top_densenet121(base_model)

        ## Densenet169
        elif self.config.model_name == 'densenet169':
            base_model = DenseNet169(weights='imagenet', include_top=False, input_tensor=self.features, input_shape=input_shape)

        ## Densenet201
        elif self.config.model_name == 'densenet201':
            base_model = DenseNet201(weights='imagenet', include_top=False, input_tensor=self.features, input_shape=input_shape)

        else:
            logging.error('Unknown model_name {}'.format(model_name))
            exit(1)

        return logits


    def model_top_densenet121(self, base_model):

        ## Freeze all densenet layers
        # base_model.trainable = False

        ## Freeze first 60 densenet layers
        # for layer in base_model.layers[:60]:
        #     layer.trainable = False

        logging.debug('base_model {}'.format(base_model.summary()))
        logging.debug('base_model layers count {}'.format(len(base_model.layers)))

        scope = 'densenet_121_top/'
        
        x = Flatten()(base_model.output)

        fc_layers = [512, 512]
        for i, n_units in enumerate(fc_layers):
            x = Dense(n_units, activation=tf.nn.relu, name=scope + 'fc_' + str(i))(x)
            x = Dropout(0.25)(x)

        x = Dense(512, name=scope + 'fc_' + str(len(fc_layers)))(x)

        x = Dense(self.config.num_classes, name=scope + 'fc_last')(x)

        logits = x

        return logits


    def model_top_resnet50(self, base_model):

        ## Freeze all layers
        # base_model.trainable = False

        ## Freeze first 60 layers
        # for layer in base_model.layers[:60]:
        #     layer.trainable = False

        logging.debug('base_model {}'.format(base_model.summary()))
        logging.debug('base_model layers count {}'.format(len(base_model.layers)))

        scope = 'resnet_50_top/'

        x = Flatten()(base_model.output)

        fc_layers = [512, 512]
        for i, n_units in enumerate(fc_layers):
            x = Dense(n_units, activation=tf.nn.relu, name=scope + 'fc_' + str(i))(x)
            x = Dropout(0.25)(x)

        x = Dense(512, name=scope + 'fc_' + str(len(fc_layers)))(x)

        x = Dense(self.config.num_classes, name=scope + 'fc_last')(x)

        logits = x

        return logits


"""
# your class weights
# class_weights = tf.constant([[1.0, 2.0, 3.0]])
# # deduce weights for batch samples based on their true label
# weights = tf.reduce_sum(class_weights * onehot_labels, axis=1)
# # apply the weights, relying on broadcasting of the multiplication
# weighted_cross_entropy = cross_entropy * weights

# # Scale the cost by the class weights
# scaled_error = tf.mul(error, class_weight)
"""
