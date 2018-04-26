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
                                            save_checkpoints_steps=5000,
                                            save_summary_steps=1000)


        ## Tensorflow Model
        # self.model_estimator = tf.estimator.Estimator(model_fn=self.model_fn_2,
        #                                               params=params,
        #                                               config=est_config,
        #                                               model_dir=self.config.checkpoint_dir)



        ## Keras Model
        # NOTE: model_dir should be absolute path
        model_dir = os.path.join(os.getcwd(), self.config.checkpoint_dir, 'keras') 
        model = self.model_fn()
        self.model_estimator = tf.keras.estimator.model_to_estimator(keras_model=model,
                                                                     config=est_config,
                                                                     custom_objects=params,
                                                                     model_dir=model_dir)


    # def model_fn(self, features, labels, mode, params, config):
    def model_fn(self):

        # TODO: _aSk Check order
        img_shape = (self.config.tfr_image_height, self.config.tfr_image_width, self.config.tfr_image_channels)


        # conv_base = VGG16(weights='imagenet', include_top=False, input_shape=img_shape)
        # model = models.Sequential()
        # model.add(conv_base)
        # model.add(layers.Flatten())

        # # model.add(layers.Dense(256, activation='relu', input_dim=self.config.tfr_image_height * self.config.tfr_image_width * self.config.tfr_image_channels))

        # model.add(layers.Dense(256, activation='relu'))
        # model.add(layers.Dense(self.config.num_classes, activation='sigmoid'))
        # conv_base.trainable = False
  	


        model = Sequential()

        # 1st layer should be named 'images' to match with {'images_input': value} fed by tf.Dataset API
        model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=img_shape, name='images'))
        model.add(Convolution2D(32, 3, 3, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.config.num_classes, activation='softmax'))

        model.summary()

        model.compile(
                loss='categorical_crossentropy',
                optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
                metrics=['acc'])

        return model


    def model_fn_2(self, features, labels, mode, params, config):
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

        # Reference to the tensor named "image" in the input-function.
        x = features['images_input']

        # 32x3072 i.e batch size of 32 with 32x32x3 image
        # if self.config.debug_tf_print:
        #     x = tf.Print(x, [x, tf.shape(x)], '\nTF x\n', summarize=10)


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

        # Softmax output of the neural network.
        y_pred = tf.nn.softmax(logits=logits)

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

        return spec

