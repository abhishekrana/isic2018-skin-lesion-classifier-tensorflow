import tensorflow as tf

def weighted_loss(logits, labels, num_classes, head=None):
    with tf.name_scope('loss_1'):
        logits = tf.reshape(logits, (-1, num_classes))
        epsilon = tf.constant(value=1e-10)
        logits = logits + epsilon
        # consturct one-hot label array
        label_flat = tf.reshape(labels, (-1, 1))
        labels = tf.reshape(tf.one_hot(label_flat, depth=num_classes), (-1, num_classes))
        labels = tf.argmax(labels)
        softmax = tf.nn.softmax(logits)
        cross_entropy = -tf.reduce_sum(tf.mul(labels * tf.log(softmax + epsilon), coefficients), reduction_indices=[1])
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)
        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        return loss
