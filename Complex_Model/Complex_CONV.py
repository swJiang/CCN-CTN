import tensorflow as tf


def complex_conv2d(inputs,
                   filters,
                   kernel_size,
                   kernal_initializer=tf.truncated_normal_initializer(stddev=0.01),
                   strides=(1, 1, 1, 1),
                   padding='SAME',
                   activation=tf.nn.relu,
                   use_bias=True,
                   bias_initializer=tf.zeros_initializer(),
                   trainable=True,
                   name=None,
                   reuse=None):
    with tf.variable_scope(name):  # , reuse=tf.AUTO_REUSE):

        real = tf.real(inputs)
        imag = tf.imag(inputs)
        W_real = tf.get_variable('W_real', dtype=tf.float32,
                                 shape=[kernel_size[0], kernel_size[1], real.shape[-1], filters],
                                 initializer=kernal_initializer)
        W_imag = tf.get_variable('W_imag', dtype=tf.float32,
                                 shape=[kernel_size[0], kernel_size[1], real.shape[-1], filters],
                                 initializer=kernal_initializer)
        b_real = tf.get_variable('b_real', dtype=tf.float32, shape=filters, initializer=bias_initializer)
        b_imag = tf.get_variable('b_imag', dtype=tf.float32, shape=filters, initializer=bias_initializer)
        conv_real = tf.nn.conv2d(real, W_real, strides, padding) - tf.nn.conv2d(imag, W_imag, strides, padding)
        conv_imag = tf.nn.conv2d(real, W_imag, strides, padding) + tf.nn.conv2d(imag, W_real, strides, padding)
        if use_bias:
            add_bias_real = tf.nn.bias_add(conv_real, b_real)
            add_bias_imag = tf.nn.bias_add(conv_imag, b_imag)
        else:
            add_bias_real = conv_real
            add_bias_imag = conv_imag
        if activation:
            return tf.complex(activation(add_bias_real), activation(add_bias_imag))
        else:
            return tf.complex(add_bias_real, add_bias_imag)
