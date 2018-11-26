import tensorflow as tf


cfg = tf.contrib.training.HParams(
    #cfg
    batch_size = 128,
    lr = 0.0001,
    decay_step = 1000,
    decay_rate = 0.96,
    epoch_num = 20
)


