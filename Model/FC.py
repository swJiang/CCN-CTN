import tensorflow as tf
def fc_layer(bottom, n_weight, name):
    assert len(bottom.get_shape()) == 2
    n_prev_weight = bottom.get_shape()[1]
    initer = tf.truncated_normal_initializer(stddev=0.01)
    W = tf.get_variable(name + 'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
    b = tf.get_variable(name + 'b', dtype=tf.float32,
                        initializer=tf.constant(0.01, shape=[n_weight], dtype=tf.float32))
    fc = tf.nn.bias_add(tf.matmul(bottom, W), b)
    return fc