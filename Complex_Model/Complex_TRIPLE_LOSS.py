import tensorflow as tf
def clip_distance(x):
    '''
    compute exp(x) and clip value in order not to exceed tf.float32 cant express
    '''
    return tf.clip_by_value(x,0.0,50.0)
def triple_loss(branch1, branch2, branch3):
    # compute p_distance match pairs distance
    p_distance = tf.sqrt(tf.reduce_sum(tf.square(branch1 - branch2), axis=1))
    # clip distance in order not qual 0.0
    p_distance = tf.clip_by_value(p_distance, 1e-2, tf.reduce_max(p_distance))
    # compute two non-match pair distance
    n1_distance = tf.sqrt(tf.reduce_sum(tf.square(branch1 - branch3), axis=1))
    n2_distance = tf.sqrt(tf.reduce_sum(tf.square(branch2 - branch3), axis=1))
    # select softnegative loss
    n_distance = tf.minimum(n1_distance, n2_distance)
    # clip distance in order not qual 0.0
    n_distance = tf.clip_by_value(n_distance, 1e-2, tf.reduce_max(n_distance))
    # compute loss
    p_loss = tf.reduce_sum(tf.square(tf.log(p_distance)))
    n_loss = tf.reduce_sum(tf.square(2 - tf.log(n_distance)))

    tf.losses.add_loss(p_loss + n_loss)
    tf.losses.add_loss(p_loss+n_loss)
    # get total loss
    total_loss = tf.losses.get_total_loss()

    return total_loss,p_loss,n_loss


def distance(o1,o2):
    distance_ = tf.sqrt(tf.reduce_sum(tf.square(o1 - o2), axis=1))
    return distance_

def complex_triple_loss(branch1, branch2, branch3):
    # compute p_distance match pairs distance
    p_distance = tf.sqrt(tf.reduce_sum(tf.square(tf.real(branch1) - tf.real(branch2)) +tf.square(tf.imag(branch1) - tf.imag(branch2)), axis=1))
    # clip distance in order not qual 0.0
    p_distance = tf.clip_by_value(p_distance, 1e-2, tf.reduce_max(p_distance))
    # compute two non-match pair distance
    n1_distance = tf.sqrt(tf.reduce_sum(tf.square(tf.real(branch1) - tf.real(branch3)) +tf.square(tf.imag(branch1) - tf.imag(branch3)), axis=1))
    n2_distance = tf.sqrt(tf.reduce_sum(tf.square(tf.real(branch2) - tf.real(branch3)) +tf.square(tf.imag(branch2) - tf.imag(branch3)), axis=1))
    # select softnegative loss
    n_distance = tf.minimum(n1_distance, n2_distance)
    # clip distance in order not qual 0.0
    n_distance = tf.clip_by_value(n_distance, 1e-2, tf.reduce_max(n_distance))
    # compute loss
    p_loss = tf.reduce_sum(tf.square(tf.log(p_distance)))
    n_loss = tf.reduce_sum(tf.square(2 - tf.log(n_distance)))

    tf.losses.add_loss(p_loss + n_loss)
    tf.losses.add_loss(p_loss+n_loss)
    # get total loss
    total_loss = tf.losses.get_total_loss()

    return total_loss,p_loss,n_loss


def complex_distance(o1,o2):
    distance_ = tf.sqrt(tf.reduce_sum(tf.square(tf.real(o1) - tf.real(o2)) +tf.square(tf.imag(o1) - tf.imag(o2)), axis=1))

    return distance_
