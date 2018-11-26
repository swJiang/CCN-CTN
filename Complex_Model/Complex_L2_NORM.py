import tensorflow as tf
def complex_l2norm(inputs):
    real = tf.nn.l2_normalize(tf.real(inputs), -1)
    imag = tf.nn.l2_normalize(tf.imag(inputs), -1)
    return tf.complex(real, imag)