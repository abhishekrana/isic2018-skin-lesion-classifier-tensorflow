from base.base_model import BaseModel
import tensorflow as tf


class MnistModel(BaseModel):
    def __init__(self, config):
        super(MnistModel, self).__init__(config)

        self.params = {"learning_rate": config.learning_rate}

        self.build_model()
        self.init_saver()

    def build_model(self):
        # here you build the tensorflow graph of any model you want and also define the loss.
        self.model_estimator = tf.estimator.Estimator(model_fn=self.model_fn, params=self.params, model_dir="temp/")


    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        pass


    def model_fn(self, features, labels, mode, params):
        # Args:
        #
        # features: This is the x-arg from the input_fn.
        # labels:   This is the y-arg from the input_fn,
        #           see e.g. train_input_fn for these two.
        # mode:     Either TRAIN, EVAL, or PREDICT
        # params:   User-defined hyper-parameters, e.g. learning-rate.

        # Reference to the tensor named "x" in the input-function.
        x = features["x"]

        # TODO: Hardcoding
        img_size = 28
        num_channels = 1

        # The convolutional layers expect 4-rank tensors
        # but x is a 2-rank tensor, so reshape it.
        net = tf.reshape(x, [-1, img_size, img_size, num_channels])


        # First convolutional layer.
        net = tf.layers.conv2d(inputs=net, name='layer_conv1',
                            filters=16, kernel_size=5,
                            padding='same', activation=tf.nn.relu)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

        # Second convolutional layer.
        net = tf.layers.conv2d(inputs=net, name='layer_conv2',
                            filters=36, kernel_size=5,
                            padding='same', activation=tf.nn.relu)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

        # Flatten to a 2-rank tensor.
        net = tf.contrib.layers.flatten(net)
        # Eventually this should be replaced with:
        # net = tf.layers.flatten(net)

        # First fully-connected / dense layer.
        # This uses the ReLU activation function.
        net = tf.layers.dense(inputs=net, name='layer_fc1',
                            units=128, activation=tf.nn.relu)

        # Second fully-connected / dense layer.
        # This is the last layer so it does not use an activation function.
        net = tf.layers.dense(inputs=net, name='layer_fc2',
                            units=10)

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
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                        logits=logits)

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



