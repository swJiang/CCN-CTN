import tensorflow as tf


def complex_fc(inputs,
               units,
               kernal_initializer=tf.truncated_normal_initializer(stddev=0.01),
               bias_initializer=tf.zeros_initializer(),
               name=None):
    with tf.variable_scope(name):
        W_real = tf.get_variable('W_real', dtype=tf.float32,
                                 shape=[inputs.shape[-1], units],
                                 initializer=kernal_initializer)
        W_imag = tf.get_variable('W_imag', dtype=tf.float32,
                                 shape=[inputs.shape[-1], units],
                                 initializer=kernal_initializer)
        b_real = tf.get_variable('b_real', dtype=tf.float32, shape=units, initializer=bias_initializer)
        b_imag = tf.get_variable('b_imag', dtype=tf.float32, shape=units, initializer=bias_initializer)
        real = tf.nn.bias_add((tf.matmul(tf.real(inputs), W_real) - tf.matmul(tf.imag(inputs), W_imag)), b_real)
        imag = tf.nn.bias_add((tf.matmul(tf.imag(inputs), W_real) + tf.matmul(tf.real(inputs), W_imag)), b_imag)
        fc = tf.complex(real, imag)
        return fc