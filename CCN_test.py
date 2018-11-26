import tensorflow as tf

from Complex_Model.Complex_Net import Complex_Model
from Model.Net import Model
from Tools.data_load import data_load
from utils.config import cfg
from utils.plot_auc import compute_valid_roc
from utils.show_progress_bar import view_bar

train_dataType = ''
test_dataType = ''
ckpt_path=''
train_images,labels, test_images, test_labels = data_load(train_dataType, test_dataType)

xs = tf.placeholder(tf.float32, [cfg.batch_size, 64,64,2])
ys = tf.placeholder(tf.float32, [cfg.batch_size, 1])


complex_nn = Complex_Model()
complex_out = complex_nn.build_net(inputs=xs)

nn = Model()
siamese_out = nn.build_net(complex_out)
nn_out = tf.layers.dense(siamese_out, units=1, activation=tf.nn.sigmoid, name='out')

acc = nn.acc(logits=nn_out, labels=ys)
loss =nn.loss(logits=nn_out, labels=ys)
sess = tf.Session()
saver = tf.train.Saver()

saver.restore(sess, ckpt_path)

labels = []
logits = []
for start, end in zip(range(0, len(test_images), cfg.batch_size),
                      range(cfg.batch_size, len(test_images), cfg.batch_size)):
    x, y = (test_images[start:end], test_labels[start:end])
    out_ = sess.run(nn_out, feed_dict={xs: x, ys: y})
    logits.extend(out_.tolist())
    labels.extend(y.tolist())
    view_bar(start / cfg.batch_size, len(test_images) / cfg.batch_size,0,0)
x, y = (test_images[-128:], test_labels[-128:])
out_ = sess.run(nn_out, feed_dict={xs: x})
logits.extend(out_.tolist()[end - len(test_images):])
labels.extend(y.tolist()[end - len(test_images):])

compute_valid_roc(labels, logits, "CCN")
