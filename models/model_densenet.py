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
import numpy as np

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
import utils.utils as utils
from functools import partial
import pickle


from tensorboard import summary as summary_lib

class ModelDensenet(BaseModel):

    def __init__(self, config):
        super(ModelDensenet, self).__init__(config)


    def build_model(self, mode, mode_ds=None):
        """
        mode: train/eval/test
        """
        assert (mode=='train') or (mode=='eval') or (mode=='predict')

        logging.debug('build_model')
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # Inputs
        self.features = tf.placeholder(tf.float32, shape=[None, self.config.tfr_image_height, self.config.tfr_image_width, self.config.tfr_image_channels], name='features_ph')
        self.labels = tf.placeholder(tf.float32, shape=[None, self.config.num_classes], name='labels_ph')


        ## Model
        self.logits = self.create_model()


        self.labels_pred_prob = tf.nn.softmax(logits=self.logits)
        self.labels_pred_cls = tf.argmax(input=self.labels_pred_prob, axis=1)

        if (mode=='train') or (mode=='eval'):

            # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
            self.cross_entropy = cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)

            # class_weights =  np.array([ 1.2755102,   0.21409838,  2.72108844,  4.53514739,  1.29282482, 12.84109149, 9.44510035])
            # weights = tf.gather(class_weights, tf.argmax(self.labels, axis=1))
            # cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=self.labels, logits=self.logits, weights=weights)


            ## Loss
            self.loss = tf.reduce_mean(cross_entropy)
            # self.loss = self.weighted_loss(self.logits, self.labels, self.config.num_classes)
            # self.loss = self.weighted_loss_v2(self.logits, self.labels, self.config.num_classes)


            ## Optimizer
            if self.config.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
            elif self.config.optimizer == 'rms_prop':
                optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config.learning_rate)
            elif self.config.optimizer == 'gradient_descent':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.learning_rate)
            elif self.config.optimizer == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate=self.config.learning_rate, momentum=self.config.momentum, use_nesterov=True)
            else:
                logging.error('Unknown optimizer {}'.format(self.config.optimizer))
                exit(1)


            ## Train op
            self.train_op = optimizer.minimize(loss=self.loss, global_step=tf.train.get_global_step())


            ## Evaluation metrics
            labels_gt = tf.argmax(self.labels, axis=1)
            self.metrics = \
            {
                "accuracy":                 tf.metrics.accuracy(labels=labels_gt, predictions=self.labels_pred_cls),
                "auc/ROC":                  tf.metrics.auc(labels=labels_gt, predictions=tf.reduce_max(self.labels_pred_prob, axis=1), num_thresholds=200, curve='ROC', summation_method='careful_interpolation'),
                "auc/PR":                   tf.metrics.auc(labels=labels_gt, predictions=tf.reduce_max(self.labels_pred_prob, axis=1), num_thresholds=200, curve='PR', summation_method='careful_interpolation'),
                "precision":                tf.metrics.precision(labels=labels_gt, predictions=self.labels_pred_cls),
                "recall":                   tf.metrics.recall(labels=labels_gt, predictions=self.labels_pred_cls),
                "true_positives":           tf.metrics.true_positives(labels=labels_gt, predictions=self.labels_pred_cls),
                "false_positives":          tf.metrics.false_positives(labels=labels_gt, predictions=self.labels_pred_cls),
                "true_negatives":           tf.metrics.true_negatives(labels=labels_gt, predictions=self.labels_pred_cls),
                "false_negatives":          tf.metrics.false_negatives(labels=labels_gt, predictions=self.labels_pred_cls)
            }


            # thresholds=np.arange(0.1, 1, 0.1).tolist()
            # tf_print_op = tf.metrics.precision_at_thresholds(labels=labels_gt, predictions=tf.reduce_max(self.labels_pred_prob, axis=1), thresholds=thresholds),
            # self.tf_print_op = tf.Print(tf_print_op, [tf_print_op, tf.shape(tf_print_op)], '\ntf_print_op\n', summarize=20, first_n=2)


            ## Histogram summary for all trainable variables (slow?)
            for v in tf.trainable_variables():
                logging.debug('{}'.format(v))
                # if 'densenet_121_top' in v.name:
                #     tf.summary.histogram(v.name, v)


        ## Save Model
        # TODO: Set up the Saver after setting up the AdamOptimizer because ADAM has state (namely per-weight learning rates) that need to be restored as well.
        self.saver = tf.train.Saver(max_to_keep=self.config.train_keep_checkpoint_max)


        if mode=='predict':
            logging.debug('mode {}'.format(mode))
            return


        ## Summaries
        tf.summary.scalar('loss', self.loss)
        # with tf.name_scope(mode_ds + '/metrics'):
        with tf.name_scope('metrics'):
            for metric_name, metric in self.metrics.items():
                m_name = metric_name + '//' + metric[0].name.replace(':', '_')
                logging.debug('m_name {}'.format(m_name))
                tf.summary.scalar(m_name , metric[0])

        # self.summary_pr = summary_lib.pr_curve('precision_recall', predictions=self.labels_pred_cls, labels=labels_gt.astype(bool), num_thresholds=21)
        # tf.summary.pr_curve('loss', tf.metrics.precision_at_thresholds(labels=labels_pr, predictions=predictions_pr, thresholds=[0.1, 0.5, 0.9]))

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
            logits = self.model_top_densenet121(base_model)

        ## Densenet201
        elif self.config.model_name == 'densenet201':
            base_model = DenseNet201(weights='imagenet', include_top=False, input_tensor=self.features, input_shape=input_shape)
            logits = self.model_top_densenet121(base_model)

        else:
            logging.error('Unknown model_name {}'.format(model_name))
            exit(1)

        return logits


    def model_top_densenet121(self, base_model):

        ## Freeze all densenet layers
        # base_model.trainable = False
        # for layer in base_model.layers:
        #     logging.debug('layer {}'.format(layer))
        #     layer.trainable = False

        ## Freeze first 60 densenet layers
        # for layer in base_model.layers[:60]:
        #     layer.trainable = False

        logging.debug('base_model {}'.format(base_model.summary()))
        logging.debug('base_model layers count {}'.format(len(base_model.layers)))

        scope = 'densenet_121_top/'

        x = Flatten()(base_model.output)

        fc_layers = [512, 512]
        for i, n_units in enumerate(fc_layers):
            # x = Dense(n_units, activation=tf.nn.tanh, name=scope + 'fc_' + str(i))(x)
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



    # def weighted_loss(logits, labels, num_classes, head=None):
    def weighted_loss(self, logits, labels, num_classes):
        with tf.name_scope('loss_1'):

            ## shape=(?, 7) dtype=float32>
            logits = tf.reshape(logits, (-1, num_classes))

            epsilon = tf.constant(value=1e-10)

            logits = logits + epsilon

            # consturct one-hot label array

            ## shape=(?, 1) dtype=float32>
            label_flat = tf.reshape(labels, (-1, 1))

            ## shape=(?, 7) dtype=float32>
            # labels = tf.reshape(tf.one_hot(label_flat, depth=num_classes), (-1, num_classes))
            labels = tf.reshape(tf.one_hot(tf.cast(label_flat, tf.int32), depth=num_classes), (-1, num_classes))

            ## shape=(?, 7) dtype=float32>
            softmax = tf.nn.softmax(logits)

            ## shape=(7, 1) dtype=float32>
            class_weights = tf.reshape(tf.convert_to_tensor(self.config.class_weights, dtype=tf.float32), (-1, 1))

            ## shape=(?,) dtype=float32>
            # shape=(?,) dtype=float32>
            cross_entropy = -tf.reduce_sum(tf.matmul(labels * tf.log(softmax + epsilon), class_weights), reduction_indices=[1])

            ## shape=() dtype=float32>
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

            tf.add_to_collection('losses', cross_entropy_mean)

            ## shape=() dtype=float32>
            # shape=() dtype=float32>
            loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

            return loss


    def weighted_loss_v2(self, logits, labels, num_classes):
        """
        labels: one hot encoded
        """

        # logits = tf.Print(logits, [logits, tf.shape(logits)], '\ntf_print_op logits\n', summarize=20, first_n=2)
        # labels = tf.Print(labels, [labels, tf.shape(labels)], '\ntf_print_op labels\n', summarize=20, first_n=2)

        # class_weights = tf.constant([[1.0, 2.0, 3.0]])
        class_weights = tf.constant([self.config.class_weights])
        # class_weights = tf.Print(class_weights, [class_weights, tf.shape(class_weights)], '\ntf_print_op class_weights\n', summarize=20, first_n=2)

        # deduce weights for batch samples based on their true label
        weights = tf.reduce_sum(class_weights * labels, axis=1)
        # weights = tf.Print(weights, [weights, tf.shape(weights)], '\ntf_print_op weights\n', summarize=20, first_n=2)

        # compute your (unweighted) softmax cross entropy loss
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        # unweighted_losses = tf.Print(unweighted_losses, [unweighted_losses, tf.shape(unweighted_losses)], '\ntf_print_op unweighted_losses\n', summarize=20, first_n=2)

        # apply the weights, relying on broadcasting of the multiplication
        weighted_losses = unweighted_losses * weights
        # weighted_losses = tf.Print(weighted_losses, [weighted_losses, tf.shape(weighted_losses)], '\ntf_print_op weighted_losses\n', summarize=20, first_n=2)

        # reduce the result to get your final loss
        loss = tf.reduce_mean(weighted_losses)
        # loss = tf.Print(loss, [loss, tf.shape(loss)], '\ntf_print_op loss\n', summarize=20, first_n=2)

        # def _debug_func(logits, class_weights, weights, unweighted_losses, weighted_losses, loss):


        def _debug_func(
            logits, labels, num_classes, class_weights, weights, unweighted_losses, weighted_losses, loss
            ):
            pu.db; return False
        debug_op = tf.py_func(func=_debug_func, Tout=[tf.bool], inp=[
            logits, labels, num_classes, class_weights, weights, unweighted_losses, weighted_losses, loss
            ])
        with tf.control_dependencies(debug_op):
            loss = tf.identity(loss)

        return loss


    # https://github.com/tkuanlun350/Tensorflow-SegNet/blob/5b52411a16ccafa020d54cf3a87e38b8db9390ae/model.py
    def weighted_loss_v3(self, logits, labels, num_classes):
        """ median-frequency re-weighting
        train_labels_node = tf.placeholder(tf.int64, shape=[batch_size, image_h, image_w, 1])
        labels: not one hot
        """

        labels = tf.argmax(input=labels, axis=1)
        labels = tf.cast(labels, tf.int32)

        ## TODO: Hardcoded
        loss_weight = np.array([ 1.2755102,   0.21409838,  2.72108844,  4.53514739,  1.29282482, 12.84109149, 9.44510035])

        with tf.name_scope('loss'):

            logits = tf.reshape(logits, (-1, num_classes))

            epsilon = tf.constant(value=1e-10)

            logits = logits + epsilon

            # consturct one-hot label array
            label_flat = tf.reshape(labels, (-1, 1))

            # should be [batch ,num_classes]
            labels = tf.reshape(tf.one_hot(label_flat, depth=num_classes), (-1, num_classes))

            softmax = tf.nn.softmax(logits)

            cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax + epsilon), loss_weight), axis=[1])

            cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

            tf.add_to_collection('losses', cross_entropy_mean)

            loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

            return loss



"""
# your class weights
# class_weights = tf.constant([[1.0, 2.0, 3.0]])
# # deduce weights for batch samples based on their true label
# weights = tf.reduce_sum(class_weights * onehot_labels, axis=1)
# # apply the weights, relying on broadcasting of the multiplication
# weighted_cross_entropy = cross_entropy * weights

# # Scale the cost by the class weights
# scaled_error = tf.mul(error, class_weight)


thresholds=np.arange(0.1, 1, 0.1).tolist()


labels_pr = tf.constant([False, True, True, False, True], dtype=tf.bool)
predictions_pr = tf.random_uniform(labels_pr.get_shape(), maxval=1.0)
self.summary_pr_op = summary_lib.pr_curve('precision_recall', predictions=predictions_pr, labels=labels_pr, num_thresholds=21)


https://wookayin.github.io/tensorflow-talk-debugging/#43
## tf.Assert
tf.Assert(condition, data, summarize=None, name=None)

# let's ensure that all the outputs in `out` are positive
assert_op = tf.Assert(tf.reduce_all(out > 0), [out], name='assert_out_positive')
with tf.control_dependencies([assert_op]):
    out = tf.identity(out, name='out')

# OR
assert_op = tf.Assert(tf.reduce_all(out > 0), [out], name='assert_out_positive')
out = tf.with_dependencies([assert_op], out)

# OR
# store all the created assertion operations into a collection, (merge them into a single op), and explicitly evaluate them using Session.run()
tf.add_to_collection('Asserts',
    tf.Assert(tf.reduce_all(out > 0), [out], name='assert_out_gt_0')
)
# merge all assertion ops from the collection
assert_op = tf.group(*tf.get_collection('Asserts'))
... = session.run([train_op, assert_op], feed_dict={...})


## tf.Print
tf.Print(input, data, message=None, first_n=None, summarize=None, name=None)
out = tf.Print(out, [tf.argmax(out, 1)], 'argmax(out) = ', summarize=20, first_n=7) # first_n: print first n elments from batch_size

g = tf.get_default_graph()
[op.name for op in g.get_operations()]


>>> g = tf.get_default_graph()
<tensorflow.python.framework.ops.Graph object at 0x7f4a2e7baf60>
>>> op = g.get_operation_by_name('densenet_121_top/fc_last/kernel/Adam_1')
<tf.Operation 'densenet_121_top/fc_last/kernel/Adam_1' type=VariableV2>
>>> op.outputs
[<tf.Tensor 'densenet_121_top/fc_last/kernel/Adam_1:0' shape=(512, 7) dtype=float32_ref>]
>>> fc = op.outputs[0]
<tf.Tensor 'densenet_121_top/fc_last/kernel/Adam_1:0' shape=(512, 7) dtype=float32_ref>
>>> self.sess.run(fc, {self.model.features: features, self.model.labels: labels_gt})
array([[5.60868472e-08, 2.35913387e-08, 3.45381288e-08, ...,
        1.19811844e-07, 3.60919721e-08, 2.93327727e-08],
       [6.02735328e-09, 8.38048830e-09, 3.07293124e-09, ...,
        2.56582644e-09, 3.24383165e-09, 2.54809751e-09],
       [3.57267993e-09, 2.33410002e-09, 2.24807686e-10, ...,
        5.06702769e-09, 2.39227971e-10, 1.88749530e-10],
       ...,
       [1.46457668e-09, 1.03210351e-09, 2.10621920e-09, ...,
        1.22624053e-08, 2.16262985e-09, 1.76716097e-09],
       [1.66430798e-08, 7.43610329e-09, 1.19885195e-08, ...,
        4.75151580e-08, 1.24432518e-08, 1.01229150e-08],
       [1.00362518e-09, 4.17425289e-10, 6.58270827e-10, ...,
        2.28573049e-09, 6.41175057e-10, 5.21723498e-10]], dtype=float32)



>>> op = g.get_tensor_by_name('densenet_121_top/fc_last/BiasAdd:0')
>>> self.sess.run(op, {self.model.features: features, self.model.labels: labels_gt})
array([[ 0.20059009,  0.5049671 , -0.33458164, -0.30335823,  0.1856931 ,
        -0.2500437 , -0.21942219],
       [ 0.20668432,  0.53186226, -0.36827788, -0.31303212,  0.19983774,
        -0.27119422, -0.2338657 ],




>>> tf.GraphKeys.
tf.GraphKeys.ACTIVATIONS                   tf.GraphKeys.GLOBAL_VARIABLES              tf.GraphKeys.MODEL_VARIABLES               tf.GraphKeys.RESOURCES                     tf.GraphKeys.TRAINABLE_VARIABLES
tf.GraphKeys.ASSET_FILEPATHS               tf.GraphKeys.INIT_OP                       tf.GraphKeys.MOVING_AVERAGE_VARIABLES      tf.GraphKeys.SAVEABLE_OBJECTS              tf.GraphKeys.TRAIN_OP
tf.GraphKeys.BIASES                        tf.GraphKeys.LOCAL_INIT_OP                 tf.GraphKeys.mro(                          tf.GraphKeys.SAVERS                        tf.GraphKeys.UPDATE_OPS
tf.GraphKeys.CONCATENATED_VARIABLES        tf.GraphKeys.LOCAL_RESOURCES               tf.GraphKeys.QUEUE_RUNNERS                 tf.GraphKeys.SUMMARIES                     tf.GraphKeys.VARIABLES
tf.GraphKeys.COND_CONTEXT                  tf.GraphKeys.LOCAL_VARIABLES               tf.GraphKeys.READY_FOR_LOCAL_INIT_OP       tf.GraphKeys.SUMMARY_OP                    tf.GraphKeys.WEIGHTS
tf.GraphKeys.EVAL_STEP                     tf.GraphKeys.LOSSES                        tf.GraphKeys.READY_OP                      tf.GraphKeys.TABLE_INITIALIZERS            tf.GraphKeys.WHILE_CONTEXT
tf.GraphKeys.GLOBAL_STEP                   tf.GraphKeys.METRIC_VARIABLES              tf.GraphKeys.REGULARIZATION_LOSSES         tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES


>>> tf.get_collection(tf.GraphKeys.LOSSES)


>>> tf.get_collection(tf.GraphKeys.SUMMARIES)
[<tf.Tensor 'loss:0' shape=() dtype=string>, <tf.Tensor 'metrics/accuracy//accuracy/value_0:0' shape=() dtype=string>, <tf.Tensor 'metrics/auc/ROC//auc/value_0:0' shape=() dtype=string>, <tf.Tensor 'metrics/auc/PR//auc_1/interpolate_pr_auc_0:0' shape=() dtype=string>, <tf.Tensor 'metrics/precision//precision/value_0:0' shape=() dtype=string>, <tf.Tensor 'metrics/recall//recall/value_0:0' shape=() dtype=string>, <tf.Tensor 'metrics/true_positives//true_positives/Identity_0:0' shape=() dtype=string>, <tf.Tensor 'metrics/false_positives//false_positives/Identity_0:0' shape=() dtype=string>, <tf.Tensor 'metrics/true_negatives//true_negatives/Identity_0:0' shape=() dtype=string>, <tf.Tensor 'metrics/false_negatives//false_negatives/Identity_0:0' shape=() dtype=string>]

>>> tf.get_collection(tf.GraphKeys.TRAIN_OP)
[<tf.Operation 'Adam' type=AssignAdd>]



To get any operations or tensors that might not be stored explicitly:
    tf.get_default_graph(): Get the current (default) graph
    G.get_operations(): List all the TF ops in the graph
    G.get_operation_by_name(name): Retrieve a specific TF op
    (Q. How to convert an operation to a tensor?)
    G.get_tensor_by_name(name): Retrieve a specific tensor
    tf.get_collection(tf.GraphKeys.~~): Get the collection of some tensors

To get variables:
    tf.get_variable_scope(): Get the current variable scope
    tf.get_variable(): Get a variable (see Sharing Variables)
    tf.trainable_variables(): List all the (trainable) variables

    [v for v in tf.all_variables() if v.name == 'fc2/weights:0'][0]


self.sess.run(tf.get_default_graph().get_tensor_by_name('densenet_121_top/fc_last/BiasAdd:0'), {self.model.features: features, self.model.labels: labels_gt})
self.sess.run(tf.get_default_graph().get_operation_by_name('densenet_121_top/fc_last/kernel/Adam_1').outputs, {self.model.features: features, self.model.labels: labels_gt})

self.sess.run(tf.get_default_graph().get_tensor_by_name(''), {self.model.features: features, self.model.labels: labels_gt})
self.sess.run(tf.get_default_graph().get_operation_by_name('').outputs, {self.model.features: features, self.model.labels: labels_gt})



loss = tf.verify_tensor_all_finite(loss, msg)
tf.add_check_numerics_ops()



# # import tensorflow as tf; sess=tf.Session()
# def _debug_func(
#         mode, features, labels, logits, labels_pred_prob, labels_pred_cls,cross_entropy
#         ):
#     pu.db; return False
# debug_op = tf.py_func(func=_debug_func, Tout=[tf.bool], inp=[
#         mode, self.features, self.labels, self.logits, self.labels_pred_prob, self.labels_pred_cls, cross_entropy
#         ])
# with tf.control_dependencies(debug_op):
#     cross_entropy = tf.identity(cross_entropy)


"""
