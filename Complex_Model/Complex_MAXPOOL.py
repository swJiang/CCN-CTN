import tensorflow as tf
def complex_max2dpool(inputs,
                      pool_size, strides,
                      padding='valid',
                      name=None):
    with tf.variable_scope(name):
        real = tf.real(inputs)
        imag = tf.imag(inputs)
        real_pool = tf.layers.max_pooling2d(real,
                                            pool_size,
                                            strides,
                                            padding=padding,
                                            name="real")

        imag_pool = tf.layers.max_pooling2d(imag,
                                            pool_size,
                                            strides,
                                            padding=padding,
                                            name="imag")

        pool = tf.complex(real_pool, imag_pool)
        return pool