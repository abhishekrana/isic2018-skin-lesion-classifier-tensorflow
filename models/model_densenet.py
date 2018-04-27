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
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras.optimizers import Adam

from models.model_densenet_121 import Densenet121
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
        # model = self.model_fn()
        # model = self.dense_model_fn()
        # model = self.dense_model_fn_v2()
        model = Densenet121(self.config, nb_dense_block=4, growth_rate=32, nb_filter=64, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4, classes=self.config.num_classes, weights_path=None)

        model.summary()


        print('LearningRate: {}'.format(self.config.learning_rate))
        model.compile(
                loss='categorical_crossentropy',
                # optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
                # optimizer=tf.keras.optimizers.RMSprop(lr=self.config.learning_rate),
                optimizer=tf.keras.optimizers.Adam(lr=self.config.learning_rate),
                # optimizer=tf.keras.optimizers.Adam(),
                # optimizer=tf.keras.optimizers.SGD(),
                metrics=['accuracy'])



        self.model_estimator = tf.keras.estimator.model_to_estimator(keras_model=model,
                                                                     config=est_config,
                                                                     custom_objects=params,
                                                                     model_dir=model_dir)

    ### DENSENET V2 ###
    def dense_block_v2(self, x, k, l):
        """
        Motivated by [12], we define H_l(·) as a composite function of three consecutive operations:
        batch normalization (BN) [14], followed by a rectified linear unit (ReLU) [6] and a 3 × 3 convolution (Conv)
        """
        t = x
        for i in range(l):
            batch_norm = BatchNormalization()(t)
            relu = Activation('relu')(batch_norm)
            # conv2d_3x3 = Convolution2D(filters=k, kernel_size=(3,3), strides=(2,2), input_shape=img_shape, name='images'))
            conv2d_3x3 = Convolution2D(filters=k, kernel_size=(3,3), use_bias=False, padding='same')(relu)
            concat = Concatenate(axis=-1)([t, conv2d_3x3])
            t = concat
        return t

    def transition_layer_v2(self, x, k):
        """
        The transition layers used in our experiments consist of a batch normalization layer and 
        an 1×1 convolutional layer followed by a 2×2 average pooling layer.

        We refer to our network with such a bottleneck layer, i.e., to the 
        BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3) version of H_l, as DenseNet-B. 
        In our experiments, we let each 1×1 convolution produce 4k feature-maps.

        We use 1×1 convolution followed by 2×2 average pooling as transition
        layers between two contiguous dense blocks
        """
        batch_norm = BatchNormalization()(x)
        # relu = Activation('relu')(batch_norm)
        conv2d_bottleneck = Convolution2D(filters=k, kernel_size=(1,1), use_bias=False, padding='same')(batch_norm)
        avg_pool2d = AveragePooling2D(pool_size=(2,2))(conv2d_bottleneck)
        return avg_pool2d

    def output_layer_v2(self, x):
        """
        At the end of the last dense block, a global average pooling is performed
        and then a softmax classifier is attached.
        """
        # batch_norm = BatchNormalization()(x)
        # relu = Activation('relu')(batch_norm)
        avg_pool2d = AveragePooling2D(pool_size=(2,2))(relu)
        flat = Flatten()(avg_pool2d)
        output = Dense(self.config.num_classes, activation='softmax')(flat)
        return output


    def dense_model_fn_v2(self):
        """
        DenseNet can have very narrow layers, e.g., k = 12. 
        """
        # TODO: _aSk Check order
        img_shape = (self.config.tfr_image_height, self.config.tfr_image_width, self.config.tfr_image_channels)

        # Growth rate/ channels
        k = 16
        l = 10      # l is layer index

        # 1st layer should be named 'images_input' to match with {'images_input': value} fed by tf.Dataset API
        x = Input(shape=img_shape, name='images_input')

        # Before entering the first dense block, a convolution with 16 (or twice the growth rate for DenseNet-BC) output channels is
        # performed on the input images. For convolutional layers with kernel size 3×3, each side of the inputs is zero-padded
        # by one pixel to keep the feature-map size fixed.
        conv2d_1 = Convolution2D(filters=k, kernel_size=(3,3), use_bias=False, padding='same')(x)

        dense_block_1 = self.dense_block_v2(conv2d_1, k, l)
        transition_layer_1 = self.transition_layer_v2(dense_block_1, k)

        dense_block_2 = self.dense_block_v2(transition_layer_1, k, l)
        transition_layer_2 = self.transition_layer_v2(dense_block_2, k)

        dense_block_3 = self.dense_block_v2(transition_layer_2, k, l)
        transition_layer_3 = self.transition_layer_v2(dense_block_3, k)

        dense_block_4 = self.dense_block_v2(transition_layer_3, k, l)
        output = self.output_layer_v2(dense_block_4)

        model = Model(inputs=[x], outputs=[output])
        model.summary()


        print('LearningRate: {}'.format(self.config.learning_rate))
        model.compile(
                loss='categorical_crossentropy',
                # optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
                # optimizer=tf.keras.optimizers.RMSprop(lr=self.config.learning_rate),
                optimizer=tf.keras.optimizers.Adam(lr=self.config.learning_rate),
                # optimizer=tf.keras.optimizers.Adam(),
                # optimizer=tf.keras.optimizers.SGD(),
                metrics=['accuracy'])

        return model


    ### DENSENET ###
    def dense_block(self, x, k, l):
        t = x
        for i in range(l):
            batch_norm = BatchNormalization()(t)
            relu = Activation('relu')(batch_norm)
            # conv2d_3x3 = Convolution2D(filters=k, kernel_size=(3,3), strides=(2,2), input_shape=img_shape, name='images'))
            conv2d_3x3 = Convolution2D(filters=k, kernel_size=(3,3), use_bias=False, padding='same')(relu)
            concat = Concatenate(axis=-1)([t, conv2d_3x3])
            t = concat
        return t

    def transition_layer(self, x, k):
        batch_norm = BatchNormalization()(x)
        relu = Activation('relu')(batch_norm)
        conv2d_bottleneck = Convolution2D(filters=k, kernel_size=(1,1), use_bias=False, padding='same')(relu)
        avg_pool2d = AveragePooling2D(pool_size=(2,2))(conv2d_bottleneck)
        return avg_pool2d

    def output_layer(self, x):
        batch_norm = BatchNormalization()(x)
        relu = Activation('relu')(batch_norm)
        avg_pool2d = AveragePooling2D(pool_size=(2,2))(relu)
        flat = Flatten()(avg_pool2d)
        output = Dense(self.config.num_classes, activation='softmax')(flat)
        return output

    def dense_model_fn(self):
        # TODO: _aSk Check order
        img_shape = (self.config.tfr_image_height, self.config.tfr_image_width, self.config.tfr_image_channels)

        # Growth rate/ channels
        k = 4
        l = 10      # l is layer index

        # 1st layer should be named 'images_input' to match with {'images_input': value} fed by tf.Dataset API
        x = Input(shape=img_shape, name='images_input')
        conv2d_1 = Convolution2D(filters=k, kernel_size=(3,3), use_bias=False, padding='same')(x)

        dense_block_1 = self.dense_block(conv2d_1, k, l)
        transition_layer_1 = self.transition_layer(dense_block_1, k)

        dense_block_2 = self.dense_block(transition_layer_1, k, l)
        transition_layer_2 = self.transition_layer(dense_block_2, k)

        dense_block_3 = self.dense_block(transition_layer_2, k, l)
        transition_layer_3 = self.transition_layer(dense_block_3, k)

        dense_block_4 = self.dense_block(transition_layer_3, k, l)
        output = self.output_layer(dense_block_4)

        model = Model(inputs=[x], outputs=[output])
        model.summary()


        print('LearningRate: {}'.format(self.config.learning_rate))
        model.compile(
                loss='categorical_crossentropy',
                # optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
                # optimizer=tf.keras.optimizers.RMSprop(lr=self.config.learning_rate),
                optimizer=tf.keras.optimizers.Adam(lr=self.config.learning_rate),
                # optimizer=tf.keras.optimizers.Adam(),
                # optimizer=tf.keras.optimizers.SGD(),
                metrics=['accuracy'])

        return model


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

        print('LearningRate: {}'.format(self.config.learning_rate))
        model.compile(
                loss='categorical_crossentropy',
                # optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
                # optimizer=tf.keras.optimizers.RMSprop(lr=self.config.learning_rate),
                # optimizer=tf.keras.optimizers.Adam(lr=self.config.learning_rate),
                optimizer=tf.keras.optimizers.Adam(),
                # optimizer=tf.keras.optimizers.SGD(),
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

