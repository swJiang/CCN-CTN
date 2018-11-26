def loss_with_spring(self):
    margin = 5.0
    labels_t = self.y_
    labels_f = tf.subtract(1.0, self.y_, name="1-yi")  # labels_ = !labels;
    eucd2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
    eucd2 = tf.reduce_sum(eucd2, 1)
    eucd = tf.sqrt(eucd2 + 1e-6, name="eucd")
    C = tf.constant(margin, name="C")
    # yi*||CNN(p1i)-CNN(p2i)||^2 + (1-yi)*max(0, C-||CNN(p1i)-CNN(p2i)||^2)
    pos = tf.multiply(labels_t, eucd2, name="yi_x_eucd2")
    # neg = tf.mul(labels_f, tf.sub(0.0,eucd2), name="yi_x_eucd2")
    # neg = tf.mul(labels_f, tf.maximum(0.0, tf.sub(C,eucd2)), name="Nyi_x_C-eucd_xx_2")
    neg = tf.multiply(labels_f, tf.pow(tf.maximum(tf.subtract(C, eucd), 0), 2), name="Nyi_x_C-eucd_xx_2")
    losses = tf.add(pos, neg, name="losses")
    loss = tf.reduce_mean(losses, name="loss")
    return loss
