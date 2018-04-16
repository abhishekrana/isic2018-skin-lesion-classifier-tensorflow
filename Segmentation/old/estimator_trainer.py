from base.base_trainer import BaseTrain
from tqdm import tqdm
import numpy as np


class EstimatorTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(EstimatorTrainer, self).__init__(sess, model, data, config, logger)

        # Estimators allows for more flexibility in data-sources and how the data is randomly shuffled and iterated
        # Create an Estimator using the DNNClassifier which assumes the class-numbers are integers
        # so we use data.train.cls instead of data.train.labels which are one-hot encoded arrays.


    def train_epoch(self):
        """
       implement the logic of epoch:
       -loop on the number of iterations in the config and call the train step
       -add any summaries you want using the summary
        """
        # pass
        x = np.array(self.data.train.images)
        y = np.array(self.data.labels)

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": x},
            y= y,
            num_epochs=None,
            shuffle=True)

        # test_input_fn = tf.estimator.inputs.numpy_input_fn(
        #         x={"x": np.array(data.test.images)},
        #         y=np.array(data.test.cls),
        #         num_epochs=1,
        #         shuffle=False)


    def train_step(self):
        """
       implement the logic of the train step
       - run the tensorflow session
       - return any metrics you need to summarize
       """
        pass
