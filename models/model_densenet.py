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
        self.logits = self.model_san()

        # probs = Activation(activation=tf.nn.softmax, name=scope + 'out_prob')(self.logits)
        # self.loss = tf.losses.softmax_cross_entropy(target, logits)


        self.labels_pred = tf.nn.softmax(logits=self.logits)
        self.labels_pred_cls = tf.argmax(input=self.labels_pred, axis=1)

        if (mode=='train') or (mode=='eval'):

            # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)

            ## Loss
            # loss = tf.losses.softmax_cross_entropy(target, logits) # san
            self.loss = tf.reduce_mean(cross_entropy)

            ## Optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
            # optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config.learning_rate)
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.learning_rate)

            ## Train op
            self.train_op = optimizer.minimize(loss=self.loss, global_step=tf.train.get_global_step())

            ## Evaluation metrics
            self.metrics = \
            {
                # "accuracy": tf.metrics.accuracy(labels, self.labels_pred_cls)
                "accuracy": tf.metrics.accuracy(tf.argmax(self.labels, axis=1), self.labels_pred_cls)
            }

            # return logits, probs, weights, loss

        ## Save Model
        # TODO: Set up the Saver after setting up the AdamOptimizer because ADAM has state (namely per-weight learning rates) that need to be restored as well.
        self.saver = tf.train.Saver(max_to_keep=self.config.train_keep_checkpoint_max)


        if mode=='predict':
            logging.debug('mode {}'.format(mode))
            return


        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy/value', self.metrics['accuracy'][0])
        tf.summary.scalar('accuracy/update_op', self.metrics['accuracy'][1])

        self.summary_op = tf.summary.merge_all()


        # self.spec = self.model_densenet121_tf()

        # ## Create model
        # # self.model = self.model_densenet121()
        # self.model = self.model_san()
        # logging.debug(self.model.summary())


        # ## Set optimizer
        # # optimizer = RMSprop(lr=2e-5)
        # # optimizer = SGD(lr=self.config.learning_rate, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5.0)
        # optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)


        # ## Compile model
        # self.model.compile(
        #         # loss='sparse_categorical_crossentropy',
        #         loss='categorical_crossentropy',
        #         optimizer=optimizer,
        #         # weighted_metrics=self.config.class_weight_dict,
        #         metrics=['accuracy']
        #         )

        # # conv_base = tf.Print(conv_base, [conv_base, tf.shape(conv_base)], '\nTF y\n', summarize=20)
        # # logging.debug('conv_base {}'.format(conv_base))
        # # logging.debug('1111111111111111{}'.format(self.model.layers[0]))


    def model_densenet121(self):

        img_size = (self.config.tfr_image_height, self.config.tfr_image_width, self.config.tfr_image_channels)

        ## VGG16
        if self.config.model_name == 'vgg16':
            conv_base = VGG16(weights='imagenet', include_top=False, input_shape=img_size)

        ## Xception
        elif self.config.model_name == 'xception':
            conv_base = Xception(weights='imagenet', include_top=False, input_shape=img_size)

        ## Resnet50
        elif self.config.model_name == 'resnet50':
            conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=img_size)

        ## InceptionResNetV2
        elif self.config.model_name == 'inception_resnet_v2':
            conv_base = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=img_size)

        ## Densenet121
        elif self.config.model_name == 'densenet121':
            conv_base = DenseNet121(weights='imagenet', include_top=False, input_shape=img_size)

        ## Densenet169
        elif self.config.model_name == 'densenet169':
            conv_base = DenseNet169(weights='imagenet', include_top=False, input_shape=img_size)

        ## Densenet201
        elif self.config.model_name == 'densenet201':
            conv_base = DenseNet201(weights='imagenet', include_top=False, input_shape=img_size)

        else:
            logging.error('Unknown model_name {}'.format(model_name))
            exit(1)

        logging.debug('conv_base {}'.format(conv_base))
        logging.debug(conv_base.summary())

        model = models.Sequential()
        model.add(conv_base)

        model.add(layers.Flatten())

        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(512, activation='relu'))

        # model.add(layers.Dense(512, activation='relu'))
        # model.add(Dropout(0.25))

        # model.add(layers.Dense(512, activation='relu'))
        # model.add(Dropout(0.25))

        # model.add(layers.Dense(512, activation='relu'))
        # model.add(Dropout(0.25))

        # model.add(layers.Dense(512, activation='relu'))
        # model.add(Dropout(0.25))

        # model.add(layers.Dense(512, activation='relu'))
        # model.add(Dropout(0.25))

        # model.add(layers.Dense(512, activation='relu'))
        # model.add(Dropout(0.25))

        model.add(layers.Dense(512))

        model.add(layers.Dense(self.config.num_classes, activation='softmax', name='class_prob'))

        conv_base.trainable = False


        return model


    # def model_san(self):
    # def model_san(self, features_dict, labels, mode, params, config):
    def model_san(self):
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

        ## InceptionResNetV2
        elif self.config.model_name == 'inception_resnet_v2':
            base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_tensor=self.features, input_shape=input_shape)

        ## Densenet121
        elif self.config.model_name == 'densenet121':
            base_model = DenseNet121(weights='imagenet', include_top=False, input_tensor=self.features, input_shape=input_shape)

        ## Densenet169
        elif self.config.model_name == 'densenet169':
            base_model = DenseNet169(weights='imagenet', include_top=False, input_tensor=self.features, input_shape=input_shape)

        ## Densenet201
        elif self.config.model_name == 'densenet201':
            base_model = DenseNet201(weights='imagenet', include_top=False, input_tensor=self.features, input_shape=input_shape)

        else:
            logging.error('Unknown model_name {}'.format(model_name))
            exit(1)


        base_model.trainable = False
        logging.debug('base_model {}'.format(base_model))

        scope = 'read_model_test/'

        x = Flatten()(base_model.output)

        fc_layers = [512, 512]
        for i, n_units in enumerate(fc_layers):
            x = Dense(n_units, activation=tf.nn.relu, name=scope + 'fc_' + str(i))(x)
            x = Dropout(0.25)(x)

        x = Dense(512, name=scope + 'fc_' + str(len(fc_layers)))(x)

        x = Dense(self.config.num_classes, name=scope + 'fc_last')(x)

        logits = x

        return logits


        # # probs = Activation(activation=tf.nn.softmax, name=scope + 'out_prob')(logits)
        # weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

        # loss = tf.losses.softmax_cross_entropy(target, logits)



        # # Softmax output of the neural network.
        # y_pred = tf.nn.softmax(logits=logits)

        # # Classification output of the neural network.
        # y_pred_cls = tf.argmax(y_pred, axis=1)

        # return logits, probs, weights, loss



    def model_densenet121_tf(self, features, labels, mode, params, config):
        """
        Args:
            features: This is the x-arg from the input_fn.
            labels:   This is the y-arg from the input_fn.
                      It is None if mode=ModeKeys.PREDICT
            mode:     Either ModeKeys.TRAIN, ModeKeys.EVAL, or ModeKeys.PREDICT
            params:   User-defined hyper-parameters, e.g. learning-rate.
                      Same as Estimator params field
            config:   Allows updating things in your model_fn based on configuration such as
                      num_ps_replicas, or model_dir
                      Same as Estimator config field

        1. The TensorFlow model, e.g. a Convolutional Neural Network.
        2. The output of the model.
        3. The loss-function used to improve the model during optimization.
        4. The optimization method.
        5. Performance metrics.
        The Estimator can be run in three modes: Training, Evaluation, or Prediction.
        The code is mostly the same, but in Prediction-mode we do not need to setup the
        loss-function and optimizer.
        """
        print('mode', mode)

        img_size = (self.config.tfr_image_height, self.config.tfr_image_width, self.config.tfr_image_channels)
        n_classes = 7
        fc_layers = [512, 512]
        scope = 'read_datasetsnet_test/'

        input_ph = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='input_img')


        # Reference to the tensor named "image" in the input-function.
        # x = features['images_input']
        # x = features['resnet50_input']
        x_key = self.config.model_name + '_input'
        x = features[x_key]


        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=img_size)
        # base_model = ResNet50(include_top=False, input_shape=img_size, input_tensor=x)
        # base_model = ResNet50(weights='imagenet', include_top=False, input_shape=img_size)
        # logging.debug('base_model {}'.format(base_model))
        # x = Flatten()(base_model.output)

        # trainabel false TOOD
        
        # The convolutional layers expect 4-rank tensors but x is a 2-rank tensor, so reshape it.
        # net = tf.reshape(x, [-1, self.config.tfr_image_height, self.config.tfr_image_width, self.config.tfr_image_channels])
        net = x

        # First convolutional layer.
        net = tf.layers.conv2d(inputs=net, name='layer_conv1',
                            filters=32, kernel_size=3,
                            padding='same', activation=tf.nn.relu)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

        # Second convolutional layer.
        net = tf.layers.conv2d(inputs=net, name='layer_conv2',
                            filters=32, kernel_size=3,
                            padding='same', activation=tf.nn.relu)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

        # Flatten to a 2-rank tensor.
        net = tf.layers.flatten(net)

        # First fully-connected / dense layer.
        # This uses the ReLU activation function.
        net = tf.layers.dense(inputs=net, name='layer_fc1',
                            units=128, activation=tf.nn.relu)

        # Second fully-connected / dense layer.
        # This is the last layer so it does not use an activation function.
        net = tf.layers.dense(inputs=net, name='layer_fc_2',
                            units=self.config.num_classes)

        # Logits output of the neural network.
        logits = net









        # layers.Input()
            #input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
        # base_model = ResNet50(include_top=False, image_shape=img_size, input_tensor=input_ph)
        # base_model = ResNet50(include_top=False, input_shape=img_size, input_tensor=x)

        # base_model = ResNet50(include_top=False, input_shape=img_size)
        # x = Flatten()(base_model.output)

        # for i, n_units in enumerate(fc_layers):
        #     x = Dense(n_units, activation=tf.nn.relu, name=scope + 'fc' + str(i))(x)

        # x = Dense(n_classes, name=scope + 'fclast')(x)
        # logits = x
        # weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)





        # Softmax output of the neural network.
        y_pred = tf.nn.softmax(logits=logits)
        # y_pred = Activation(activation=tf.nn.softmax, name=scope + 'out_prob')(logits)

        # Classification output of the neural network.
        y_pred_cls = tf.argmax(y_pred, axis=1)

        if mode == tf.estimator.ModeKeys.PREDICT:
            # If the estimator is supposed to be in prediction-mode
            # then use the predicted class-number that is output by
            # the neural network. Optimization etc. is not needed.
            spec = tf.estimator.EstimatorSpec(mode=mode,
                                            predictions=y_pred_cls)

        else:
            # Otherwise the estimator is supposed to be in either
            # training or evaluation-mode. Note that the loss-function
            # is also required in Evaluation mode.

            # Define the loss-function to be optimized, by first
            # calculating the cross-entropy between the output of
            # the neural network and the true labels for the input data.
            # This gives the cross-entropy for each image in the batch.

            # Labels used in softmax_cross_entropy_with_logits are the one hot version of labels 
            # used in sparse_softmax_cross_entropy_with_logits.
            # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)


            # Reduce the cross-entropy batch-tensor to a single number
            # which can be used in optimization of the neural network.
            loss = tf.reduce_mean(cross_entropy)

            # SAN
            # loss = tf.losses.softmax_cross_entropy(target, logits)

            # Define the optimizer for improving the neural network.
            optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])

            # Get the TensorFlow op for doing a single optimization step.
            train_op = optimizer.minimize(
                loss=loss, global_step=tf.train.get_global_step())

            # Define the evaluation metrics,
            # in this case the classification accuracy.
            metrics = \
            {
                # "accuracy": tf.metrics.accuracy(labels, y_pred_cls)
                "accuracy": tf.metrics.accuracy(tf.argmax(labels, axis=1), y_pred_cls)
            }

            # Wrap all of this in an EstimatorSpec.
            spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                eval_metric_ops=metrics)

        # return logits, probs, weights, loss
        return spec

