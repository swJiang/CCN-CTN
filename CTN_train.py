import numpy as np
import tensorflow as tf

from Complex_Model.Complex_Net import Complex_Model
from Tools.data_load import triple_data_load
from utils.config import cfg
from utils.plot_auc import compute_valid_roc
from utils.show_progress_bar import view_bar

train_dataType = ''
test_dataType = ''
ckpt_path = ''
train_images, test_images, test_labels = triple_data_load(train_dataType, test_dataType)
x1 = tf.placeholder(tf.float32, [cfg.batch_size, 64,64,1])
x2 = tf.placeholder(tf.float32, [cfg.batch_size, 64,64,1])
x3 = tf.placeholder(tf.float32, [cfg.batch_size, 64,64,1])
# xs_ = tf.complex(xs, 0.0)

complex_nn = Complex_Model()
with tf.variable_scope("triple") as scope:
    complex_out_1 = complex_nn.build_net(x1)
    complex_out_1 = complex_nn.complex_l2norm(complex_out_1)
    scope.reuse_variables()
    complex_out_2 = complex_nn.build_net(x2)
    complex_out_2 = complex_nn.complex_l2norm(complex_out_2)
    complex_out_3 = complex_nn.build_net(x3)
    complex_out_3 = complex_nn.complex_l2norm(complex_out_3)

distance = complex_nn.complex_distance(complex_out_1, complex_out_2)
total_loss, p_loss, n_loss  = complex_nn.complex_triple_loss(complex_out_1, complex_out_2,complex_out_3)


global_step = tf.Variable(0,trainable=False)
learing_rate = tf.train.exponential_decay(0.001,global_step,cfg.decay_step,cfg.decay_rate,staircase=False)
optimizer = tf.train.AdamOptimizer(learing_rate)
train_step = optimizer.minimize(total_loss)
sess = tf.Session()
saver = tf.train.Saver(max_to_keep=5)
init = tf.global_variables_initializer()

sess.run(init)
for epoch in range(cfg.epoch_num):
    tr_loss = []
    neg_loss = []
    pos_loss = []
    tr_acc = []
    for start, end in zip(range(0, len(train_images), cfg.batch_size),
                          range(cfg.batch_size, len(train_images), cfg.batch_size)):
        rand = np.random.choice(len(train_images), cfg.batch_size)
        x_1, x_2, x_3 = np.split(train_images[rand],3,-1)
        _, _loss, n_l, p_l = sess.run([train_step, total_loss, n_loss, p_loss], feed_dict={x1: x_1,x2:x_2, x3:x_3})
        # if (start/cfg.batch_size) %100 == 0:
        #     print(_loss)
        tr_loss.append(_loss)
        neg_loss.append(n_l)
        pos_loss.append(p_l)
        view_bar(start / cfg.batch_size, len(train_images) / cfg.batch_size, epoch, cfg.epoch_num)


    print("\n")
    print('epoch %d: total_loss %.6f, neg_loss %.6f, pos_loss %.6f' % (epoch, np.mean(tr_loss), np.mean(neg_loss), np.mean(pos_loss)))
    dis_ = []
    y_ = []
    for start, end in zip(range(0, len(test_images), cfg.batch_size),
                          range(cfg.batch_size, len(test_images), cfg.batch_size)):
        x, y = (test_images[start: end], test_labels[start: end])
        x_1, x_2 = np.split(x,2,-1)
        dis = sess.run(distance, feed_dict={x1: x_1, x2 : x_2})
        dis_.extend(dis.tolist())
        y_.extend(y.tolist())
    x, y = (test_images[-128:], test_labels[-128:])
    x_1, x_2 = np.split(x, 2, -1)
    dis = sess.run(distance, feed_dict={x1: x_1, x2: x_2})
    dis_.extend(dis.tolist()[end - len(test_images):])
    y_.extend(y.tolist()[end - len(test_images):])
    compute_valid_roc(y_, dis_)

    saver.save(sess,ckpt_path, global_step=epoch)