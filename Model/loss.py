import tensorflow as tf

def compute_loss(labels, logits):
    return tf.reduce_mean(tf.square(labels - logits))