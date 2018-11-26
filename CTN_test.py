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

complex_nn = Complex_Model()
with tf.variable_scope("triple") as scope:
    complex_out_1 = complex_nn.build_net(x1)
    complex_out_1 = complex_nn.complex_l2norm(complex_out_1)
    scope.reuse_variables()
    complex_out_2 = complex_nn.build_net(x2)
    complex_out_2 = complex_nn.complex_l2norm(complex_out_2)


distance = complex_nn.complex_distance(complex_out_1, complex_out_2)


sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, ckpt_path)

labels = []
logits = []
for start, end in zip(range(0, len(test_images), cfg.batch_size),
                      range(cfg.batch_size, len(test_images), cfg.batch_size)):
    x, y = (test_images[start:end], test_labels[start:end])
    x_1, x_2= np.split(x,2,-1)
    out_ = sess.run(distance, feed_dict={x1: x_1, x2: x_2})
    logits.extend(out_.tolist())
    labels.extend(y.tolist())
    view_bar(start / cfg.batch_size, len(test_images) / cfg.batch_size,0,0)
x, y = (test_images[-128:], test_labels[-128:])
x_1, x_2= np.split(x, 2, -1)
out_ = sess.run(distance, feed_dict={x1: x_1, x2: x_2})
logits.extend(out_.tolist()[end - len(test_images):])
labels.extend(y.tolist()[end - len(test_images):])

compute_valid_roc(labels, logits)
