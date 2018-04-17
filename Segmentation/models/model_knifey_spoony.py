##########################################################################################
# Model
##########################################################################################
from base.base_model import BaseModel
import tensorflow as tf
import os

class ModelKnifeySpoony(BaseModel):
    def __init__(self, config):
        super(ModelKnifeySpoony, self).__init__(config)


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

        # The convolutional layers expect 4-rank tensors but x is a 2-rank tensor, so reshape it.
        net = tf.reshape(x, [-1, self.config.tfr_image_height, self.config.tfr_image_width, self.config.tfr_image_channels])

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

