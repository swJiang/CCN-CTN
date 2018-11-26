import tensorflow as tf
def build_block(nn, inputs,
                isTraining=True,
                filters=32,
                kenel_size=[3, 3],
                pool=False,
                name=None):
    with tf.variable_scope(name):  # , reuse=True):
        W_real = tf.get_variable('W_real', dtype=tf.float32,
                                 shape=[inputs.shape[0], inputs.shape[1], inputs.shape[-1], filters],
                                 initializer=tf.truncated_normal_initializer(stddev=0.01))
        W_imag = tf.get_variable('W_imag', dtype=tf.float32,
                                 shape=[inputs.shape[0], inputs.shape[1], inputs.shape[-1], filters],
                                 initializer=tf.truncated_normal_initializer(stddev=0.01))
        bn_1 = nn.complex_BatchNormalization(inputs, isTraining, name='bn1')
        relu_1 = nn.complex_relu(bn_1)
        conv1 = nn.complex_conv2d(relu_1,
                                    filters=filters,
                                    kernel_size=kenel_size,
                                    activation=None,
                                    name="conv1")
        bn_2 = nn.complex_BatchNormalization(conv1, isTraining, name='bn2')
        relu_2 = nn.complex_relu(bn_2)
        conv2 = nn.complex_conv2d(relu_2,
                                    filters=filters,
                                    kernel_size=kenel_size,
                                    activation=None,
                                    name="conv2")
        out = tf.add(conv2, tf.matmul(inputs, tf.complex(W_real, W_imag)))

        if pool:
            out = nn.complex_max2dpool(out,
                                         pool_size=[2, 2],
                                         strides=(2, 2),
                                         name="pool")
    return out