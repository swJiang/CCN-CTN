import tensorflow as tf
def complex_relu(inputs):
    return tf.complex(tf.nn.relu(tf.real(inputs)), tf.nn.relu(tf.imag(inputs)))