import numpy as np
import tensorflow as tf

from Complex_Model.Complex_Net import Complex_Model
from Model.Net import Model
from Tools.data_load import data_load
from utils.config import cfg
from utils.show_progress_bar import view_bar

train_dataType = ''
test_dataType = ''
ckpt_path=''

train_images,labels, test_images, test_labels = data_load(train_dataType, test_dataType)
xs = tf.placeholder(tf.float32, [cfg.batch_size, 64,64,2])
ys = tf.placeholder(tf.float32, [cfg.batch_size, 1])
# xs_ = tf.complex(xs, 0.0)

complex_nn = Complex_Model()
complex_out = complex_nn.build_net(inputs=xs)

nn = Model()
siamese_out = nn.build_net(complex_out)
nn_out = tf.layers.dense(siamese_out, units=1, activation=tf.nn.sigmoid, name='out')

global_step = tf.Variable(0,trainable=False)
learing_rate = tf.train.exponential_decay(cfg.lr,global_step,cfg.decay_step,1,staircase=False)
acc = nn.acc(logits=nn_out, labels=ys)
loss =nn.loss(logits=nn_out, labels=ys)
optimizer = tf.train.AdamOptimizer(learing_rate)
train_step = optimizer.minimize(loss)
sess = tf.Session()
saver = tf.train.Saver(max_to_keep=20)
init = tf.global_variables_initializer()

sess.run(init)
for epoch in range(cfg.epoch_num):
    tr_loss = []
    tr_acc = []
    for start, end in zip(range(0, len(train_images), cfg.batch_size),
                          range(cfg.batch_size, len(train_images), cfg.batch_size)):
        rand = np.random.choice(len(train_images), cfg.batch_size)
        x, y = (train_images[rand], labels[rand])
        _, _loss, _acc = sess.run([train_step, loss, acc], feed_dict={xs: x, ys: y})
        # if (start/cfg.batch_size) %100 == 0:
        #     print(_loss, _acc)
        tr_loss.append(_loss)
        tr_acc.append(_acc)
        view_bar(start / cfg.batch_size, len(train_images) / cfg.batch_size, epoch, cfg.epoch_num)


    print("\n")
    print('epoch %d: acc %.3f sigmoid_loss %.6f' % (epoch, np.mean(tr_acc), np.mean(tr_loss)))
    te_loss = []
    te_acc = []
    for start, end in zip(range(0, len(test_images), cfg.batch_size),
                          range(cfg.batch_size, len(test_images), cfg.batch_size)):
        x, y = (test_images[start:end], test_labels[start:end])
        _loss, _acc = sess.run([loss, acc], feed_dict={xs: x, ys: y})
        te_loss.append(_loss)
        te_acc.append(_acc)
    print('test : acc %.3f sigmoid_loss %.6f' % (np.mean(te_acc), np.mean(te_loss)))

    saver.save(sess, ckpt_path, global_step=epoch)



