import tensorflow as tf
from Model.FC import fc_layer
from Model.NetWork import network
from Model.accuracy import compute_acc
from Model.loss import compute_loss
class Model(object):
    def __init__(self):
        self.FC = fc_layer
        self.network = network
        self.compute_acc = compute_acc
        self.compute_loss = compute_loss


    def build_net(self, inputs):
        with tf.variable_scope("siamese") as scope:
            o1 = self.network(self, tf.real(inputs))
            scope.reuse_variables()
            o2 = self.network(self, tf.imag(inputs))
        return tf.concat([o1,o2],axis=-1)

    def acc(self, logits, labels):
        logits = tf.round(logits)
        return self.compute_acc(logits, labels)

    def loss(self, logits, labels):
        loss = self.compute_loss(labels=labels,logits=logits)
        return loss