import tensorflow as tf
from .Complex_BN import complex_BatchNormalization
from .Complex_CONV import complex_conv2d
from .Complex_FC import complex_fc
from .Complex_L2_NORM import complex_l2norm
from .Complex_MAXPOOL import complex_max2dpool
from .Complex_RELU import complex_relu
from .Complex_TRIPLE_LOSS import triple_loss, distance, complex_triple_loss, complex_distance
from .Complex_BLOCK import build_block
from utils.config import cfg
class Complex_Model(object):
    def __init__(self, isTraining = True):
        self.complex_BatchNormalization = complex_BatchNormalization
        self.complex_conv2d = complex_conv2d
        self.complex_fc = complex_fc
        self.complex_l2norm = complex_l2norm
        self.complex_relu = complex_relu
        self.complex_max2dpool = complex_max2dpool
        self.complex_triple_loss = complex_triple_loss
        self.complex_distance = complex_distance
        self.triple_loss = triple_loss
        self.build_block = build_block
        self.isTraining = isTraining
        self.distance = distance
    def build_net(self, inputs):
        conv = tf.layers.conv2d(inputs,
                                filters=32,
                                kernel_size=3,
                                padding='same',
                                name = 'conv')

        # b0 = tf.layers.batch_normalization(conv)
        conv_out = tf.complex(conv, 0.0)
        b1 = self.build_block(self, conv_out,  filters=32, name='b1', isTraining = self.isTraining ,pool=True)
        # b2 = self.build_block(self, b1, filters=32, name='b2', isTraining = self.isTraining, pool=True)
        b3 = self.build_block(self, b1, filters=64, name='b3', isTraining = self.isTraining, pool=True)
        # b4 = self.build_block(self, b3, filters=64, name='b4', isTraining = self.isTraining, pool=True)
        # b5 = self.build_block(self, b4, filters=128, name='b5', isTraining = self.isTraining, pool=True)
        b6 = self.build_block(self, b3, filters=128, name='b6', isTraining = self.isTraining, pool=True)
        out = tf.reshape(b6, shape=[cfg.batch_size, -1])
        # out = complex_fc(complex_out,
        #                  units=128,
        #                  name= "out")
        return out

    def build_2_channel_siamese_net(self,inputs):
        conv = tf.layers.conv2d(inputs,
                                filters=32,
                                kernel_size=3,
                                padding='same',
                                name='conv')

        b0 = tf.layers.batch_normalization(conv,
                                           name='bn1')
        conv_out = tf.complex(b0, 0.0)
        b1 = self.build_block(self, conv_out, filters=32, name='b1', isTraining=self.isTraining, pool=True)
        b2 = self.build_block(self, b1, filters=32, name='b3', isTraining=self.isTraining, pool=True)
        b3 = self.build_block(self, b2, filters=32, name='b6', isTraining=self.isTraining, pool=True)
        b3_value = tf.concat([tf.real(b3), tf.imag(b3)], axis=-1)
        b4 = tf.layers.conv2d(b3_value,
                              filters=64,
                              kernel_size=3,
                              padding='same',
                              name='conv4',
                              activation=tf.nn.relu
                              )
        b4_bn = tf.layers.batch_normalization(b4,
                                              momentum=0.9,
                                              name='bn4')
        b4_pool = tf.layers.max_pooling2d(b4_bn,
                                         pool_size=[2, 2],
                                         strides=(2, 2),
                                         name="pool1")
        b5 = tf.layers.conv2d(b4_pool,
                              filters=128,
                              kernel_size=3,
                              padding='same',
                              name='conv5',
                              activation=tf.nn.relu
                              )
        b5_bn = tf.layers.batch_normalization(b5,
                                              momentum=0.9,
                                              name='bn5')
        b5_pool = tf.layers.max_pooling2d(b5_bn,
                                         pool_size=[2, 2],
                                         strides=(2, 2),
                                         name="pool2")
        b6 = tf.layers.conv2d(b5_pool,
                              filters=128,
                              kernel_size=3,
                              padding='same',
                              name='conv6',
                              activation=tf.nn.relu
                              )
        b6_bn = tf.layers.batch_normalization(b6,
                                              momentum=0.9,
                                              name='bn6')
        b6_pool = tf.layers.max_pooling2d(b6_bn,
                                         pool_size=[2, 2],
                                         strides=(2, 2),
                                         name="pool3")
        b6_reshape = tf.reshape(b6_pool, shape=[cfg.batch_size, -1])
        fc1 = tf.layers.dense(b6_reshape,
                              units=4096,
                              activation=tf.nn.relu,
                              name="fc1")
        fc2 = tf.layers.dense(fc1,
                              units=256,
                              name="fc2")
        return fc2
