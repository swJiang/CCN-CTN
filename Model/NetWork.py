import tensorflow as tf


def network(nn, x):
    x = tf.nn.l2_normalize(x)
    fc1 = nn.FC(x, 4096, "fc1")
    fc1_bn = tf.layers.batch_normalization(fc1,momentum=0.8,name='fc1_bn')
    ac1 = tf.nn.relu(fc1_bn)
    # ac1 = tf.nn.relu(fc1)
    fc2 = nn.FC(ac1, 4096, "fc2")
    fc2_bn = tf.layers.batch_normalization(fc2,momentum=0.8,name='fc2_bn')
    ac2 = tf.nn.relu(fc2_bn + ac1)
    # ac2 = tf.nn.relu(fc2+ac1)

    fc3 = nn.FC(ac2, 1024, "fc3")
    fc3_bn = tf.layers.batch_normalization(fc3,momentum=0.8,name='fc3_bn')
    ac3 = tf.nn.relu(fc3_bn)
    # ac3 = tf.nn.relu(fc3)
    return ac3