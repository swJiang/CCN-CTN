import tensorflow as tf
def compute_acc(logits, labels):
    correct_prediction = tf.cast(tf.equal(logits, labels), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    return accuracy