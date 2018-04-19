##########################################################################################
# Model
##########################################################################################
import os
os.sys.path.append('./')

from base.base_model import BaseModel
import tensorflow as tf
import random
import numpy as np

import utils.utils as utils
import utils.utils_image as utils_image
from utils.config import process_config

class ModelUnet(BaseModel):
    def __init__(self, config):
        super(ModelUnet, self).__init__(config)


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
        self.model_estimator = tf.estimator.Estimator(model_fn=self.model_fn,
                                                      params=params,
                                                      # config=config,
                                                      model_dir=self.config.checkpoint_dir)

    def tf_process_image_fn(self, image):
        pu.db
        print('image', image.shape)
        # image = tf.reshape(tf.cast(image[0], tf.uint8), [16, 16, 1])
        # image = tf.reshape(tf.cast(image[0], tf.uint8), [self.config.tfr_image_height,
        #                                                  self.config.tfr_image_width,
        #                                                  self.config.tfr_image_channels])
        # image = tf.cast(image, tf.uint8)

        image = tf.image.convert_image_dtype(image, tf.uint8)
        print('image', image.shape)
        # Encode
        images_encode = tf.image.encode_jpeg(image)

        # Create a files name
        # fname = tf.constant('datetime.now() + ".jpeg"')
        fname = tf.constant('abc.jpeg')

        # Write files
        fwrite = tf.write_file(fname, images_encode)


    def model_fn(self, features, labels, mode, params, config):
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
        print('params', params)
        print('config', config)

        # Reference to the tensor named "image" in the input-function.
        x = features["image"]
        # gt_image = labels
        pu.db
        # gt_image = tf.reshape(labels, [-1, self.config.tfr_gt_image_height, self.config.tfr_gt_image_width, 1])
        gt_image = tf.reshape(labels, [-1, self.config.tfr_gt_image_height, self.config.tfr_gt_image_width])


        # TODO: _aSk, Check if this is correct
        # https://hackernoon.com/how-tensorflows-tf-image-resize-stole-60-days-of-my-life-aba5eb093f35
        # gt_image_new = tf.image.resize_images(images=gt_image, size=[388, 388], align_corners=False)




        # The convolutional layers expect 4-rank tensors but x is a 2-rank tensor, so reshape it.
        net = tf.reshape(x, [-1, self.config.tfr_image_height, self.config.tfr_image_width, self.config.tfr_image_channels])



        tf.map_fn(self.tf_process_image_fn, net, dtype=tf.float32)


        # # Convolutional Layer #1
        # conv1 = tf.layers.conv2d(
        #             inputs=input_layer,
        #             filters=32,
        #             kernel_size=[5, 5],
        #             padding="same",
        #             activation=tf.nn.relu)

        # # Dense Layer
        # pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        # dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        # dropout = tf.layers.dropout(
        #             inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)


        # # Pooling Layer #1
        # pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)








        # # First convolutional layer.
        # net = tf.layers.conv2d(inputs=net, name='layer_conv1',
        #                     filters=32, kernel_size=3,
        #                     padding='same', activation=tf.nn.relu)
        # net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

        # # Second convolutional layer.
        # net = tf.layers.conv2d(inputs=net, name='layer_conv2',
        #                     filters=32, kernel_size=3,
        #                     padding='same', activation=tf.nn.relu)
        # net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)







        ## CONV_1 ##
        # 572x572x3
        net = tf.layers.conv2d(inputs=net, name='layer_conv_1_1',
                               filters=64, kernel_size=[3,3], padding='valid',
                               activation=tf.nn.relu)
        # 570x570x64
        net = tf.layers.conv2d(inputs=net, name='layer_conv_1_2',
                               filters=64, kernel_size=[3,3], padding='valid',
                               activation=tf.nn.relu)
        # 568x568x64
        net = tf.layers.max_pooling2d(inputs=net, name='layer_pool_1',
                                      pool_size=2, strides=2)


        ## CONV_2 ##
        # 284x284x
        net = tf.layers.conv2d(inputs=net, name='layer_conv_2_1',
                               filters=128, kernel_size=[3,3], padding='valid',
                               activation=tf.nn.relu)
        # 282x282x128
        net = tf.layers.conv2d(inputs=net, name='layer_conv_2_2',
                               filters=128, kernel_size=[3,3], padding='valid',
                               activation=tf.nn.relu)
        # 280x280x128
        net = tf.layers.max_pooling2d(inputs=net, name='layer_pool_2',
                                      pool_size=2, strides=2)


        ## CONV_3 ##
        # 140x140x
        net = tf.layers.conv2d(inputs=net, name='layer_conv_3_1',
                               filters=256, kernel_size=[3,3], padding='valid',
                               activation=tf.nn.relu)
        # 138x138x256
        net = tf.layers.conv2d(inputs=net, name='layer_conv_3_2',
                               filters=256, kernel_size=[3,3], padding='valid',
                               activation=tf.nn.relu)
        # 136x136x256
        net = tf.layers.max_pooling2d(inputs=net, name='layer_pool_3',
                                      pool_size=2, strides=2)

        ## CONV_4 ##
        # 68x68x
        net = tf.layers.conv2d(inputs=net, name='layer_conv_4_1',
                               filters=512, kernel_size=[3,3], padding='valid',
                               activation=tf.nn.relu)
        # 66x66x512
        net = tf.layers.conv2d(inputs=net, name='layer_conv_4_2',
                               filters=512, kernel_size=[3,3], padding='valid',
                               activation=tf.nn.relu)
        # 64x64x512
        net = tf.layers.max_pooling2d(inputs=net, name='layer_pool_4',
                                      pool_size=2, strides=2)

        ## CONV_5 ##
        # 32x32x
        net = tf.layers.conv2d(inputs=net, name='layer_conv_5_1',
                               filters=1024, kernel_size=[3,3], padding='valid',
                               activation=tf.nn.relu)
        # 30x30x1024
        net = tf.layers.conv2d(inputs=net, name='layer_conv_5_2',
                               filters=1024, kernel_size=[3,3], padding='valid',
                               activation=tf.nn.relu)

        ## Upconv ##

        # 28x28x1024
        net = tf.layers.conv2d_transpose(inputs=net, name='layer_conv_transpose_5_2_EXP',
                               filters=512, kernel_size=[2,2], padding='valid', strides=(2, 2))


        # 56x56x{512}
        # TODO: Concat
        # 56x56x1024
        net = tf.layers.conv2d(inputs=net, name='layer_conv_4_2_EXP',
                               filters=512, kernel_size=[3,3], padding='valid',
                               activation=tf.nn.relu)
        # 54x54x512
        net = tf.layers.conv2d(inputs=net, name='layer_conv_4_1_EXP',
                               filters=512, kernel_size=[3,3], padding='valid',
                               activation=tf.nn.relu)
        # 52x52x{512}
        net = tf.layers.conv2d_transpose(inputs=net, name='layer_conv_transpose_4_1_EXP',
                               filters=256, kernel_size=[2,2], padding='valid', strides=(2, 2))


        # 104x104x256
        # TODO: concat
        # 104x104x512
        net = tf.layers.conv2d(inputs=net, name='layer_conv_3_2_EXP',
                               filters=256, kernel_size=[3,3], padding='valid',
                               activation=tf.nn.relu)
        # 102x102x256
        net = tf.layers.conv2d(inputs=net, name='layer_conv_3_1_EXP',
                               filters=256, kernel_size=[3,3], padding='valid',
                               activation=tf.nn.relu)
        # 100x100x{256}
        net = tf.layers.conv2d_transpose(inputs=net, name='layer_conv_transpose_3_1_EXP',
                               filters=128, kernel_size=[2,2], padding='valid', strides=(2, 2))


        # 200x200x128
        # TODO: concat
        # 200x200x256
        net = tf.layers.conv2d(inputs=net, name='layer_conv_2_2_EXP',
                               filters=128, kernel_size=[3,3], padding='valid',
                               activation=tf.nn.relu)
        # 198x198x128
        net = tf.layers.conv2d(inputs=net, name='layer_conv_2_1_EXP',
                               filters=128, kernel_size=[3,3], padding='valid',
                               activation=tf.nn.relu)
        # 196x196x{128}
        net = tf.layers.conv2d_transpose(inputs=net, name='layer_conv_transpose_2_1_EXP',
                               filters=64, kernel_size=[2,2], padding='valid', strides=(2, 2))


        # 392x392x{64}
        # TODO: concat
        # 392x392x{128}
        net = tf.layers.conv2d(inputs=net, name='layer_conv_1_3_EXP',
                               filters=64, kernel_size=[3,3], padding='valid',
                               activation=tf.nn.relu)
        # 390x390x64
        net = tf.layers.conv2d(inputs=net, name='layer_conv_1_2_EXP',
                               filters=64, kernel_size=[3,3], padding='valid',
                               activation=tf.nn.relu)
        # 388x388x64
        net = tf.layers.conv2d(inputs=net, name='layer_conv_1_1_EXP',
                               filters=2, kernel_size=[1,1], padding='valid')

        # 388x388x2



        # Logits output of the neural network.
        logits = net

        pu.db



        # # Flatten to a 2-rank tensor.
        # net = tf.layers.flatten(net)

        # # First fully-connected / dense layer.
        # # This uses the ReLU activation function.
        # net = tf.layers.dense(inputs=net, name='layer_fc1',
        #                     units=128, activation=tf.nn.relu)

        # # Second fully-connected / dense layer.
        # # This is the last layer so it does not use an activation function.
        # net = tf.layers.dense(inputs=net, name='layer_fc_2',
        #                     units=self.config.num_classes)

        # # Logits output of the neural network.
        # logits = net

        # # Softmax output of the neural network.
        # y_pred = tf.nn.softmax(logits=logits)

        # # Classification output of the neural network.
        # y_pred_cls = tf.argmax(y_pred, axis=1)


        # TODO _aSk
        y_pred_cls = tf.argmax(logits, axis=3)


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

            # For sparse_softmax_cross_entropy_with_logits, labels must have the shape [batch_size] and the dtype int32 or int64. Each label is an int in range [0, num_classes-1].
            # For softmax_cross_entropy_with_logits, labels must have the shape [batch_size, num_classes] and dtype float32 or float64.
            # Labels used in softmax_cross_entropy_with_logits are the one hot version of labels used in sparse_softmax_cross_entropy_with_logits.
            # With sparse_softmax_cross_entropy_with_logits, you can give -1 as a label to have loss 0 on this label.

            # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_image, logits=logits)

            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=gt_image, logits=y_pred_cls)



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
                "accuracy": tf.metrics.accuracy(labels, y_pred_cls)
            }

            # Wrap all of this in an EstimatorSpec.
            spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                eval_metric_ops=metrics)

        return spec


if __name__ == '__main__':
    args = utils.get_args()
    config = process_config(args.config)

    img_np = np.random.rand(config.tfr_image_height * config.tfr_image_width * config.tfr_image_channels) * 255
    img_np = np.expand_dims(img_np, axis=0)
    print('img_np', img_np.shape)

    x = {'image': img_np}
    y = img_np

    labels=[1]
    params = {"learning_rate": config.learning_rate}

    model = ModelUnet(config)
    model.model_fn(features=(x, y), labels=labels, mode=tf.estimator.ModeKeys.TRAIN, params=params, config=None)






