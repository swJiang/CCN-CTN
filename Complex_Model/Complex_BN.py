import tensorflow as tf
def complex_BatchNormalization(inputs,
                               is_training,
                               name='BatchNorm',
                               moving_decay=0.9,
                               eps=1e-5 + 1e-5j):
    shape = inputs.shape
    assert len(shape) in [2, 4]
    param_shape = shape[-1]
    with tf.variable_scope(name):  # , reuse=tf.AUTO_REUSE):
        beta_real = tf.get_variable('beat_r', param_shape, initializer=tf.constant_initializer(0))
        beta_imag = tf.get_variable('beat_i', param_shape, initializer=tf.constant_initializer(0))
        gamma_real = tf.get_variable('gamma_r', param_shape, initializer=tf.constant_initializer(1))
        gamma_imag = tf.get_variable('gamma_i', param_shape, initializer=tf.constant_initializer(1))
        beta = tf.complex(beta_real, beta_imag)
        gamma = tf.complex(gamma_real, gamma_imag)

        axes = list(range(len(shape) - 1))
        batch_mean_real, batch_var_real = tf.nn.moments(tf.real(inputs), axes, name='moments')
        batch_mean_imag, batch_var_imag = tf.nn.moments(tf.imag(inputs), axes, name='moments')

        # 采用滑动平均更新均值与方差
        ema = tf.train.ExponentialMovingAverage(moving_decay)

        def mean_var_with_update():
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                ema_apply_op = ema.apply([batch_mean_real, batch_mean_imag, batch_var_real, batch_var_imag])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean_real), tf.identity(batch_mean_imag), tf.identity(
                        batch_var_real), tf.identity(batch_var_imag)

        # 训练时，更新均值与方差，测试时使用之前最后一次保存的均值与方差
        mean_real, mean_imag, var_real, var_imag = tf.cond(tf.equal(is_training, True), mean_var_with_update,
                                                           lambda: (
                                                           ema.average(batch_mean_real), ema.average(batch_mean_imag),
                                                           ema.average(batch_var_real), ema.average(batch_var_imag)))
        mean = tf.complex(mean_real, mean_imag)
        var = tf.complex(var_real, var_imag)
        # 最后执行batch normalization
        bn = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, eps)
        return bn