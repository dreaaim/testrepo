import os
import numpy as np
import tensorflow as tf
import time
#import layers
import batch_input
import test
import preprocess

flags = tf.flags
flags.DEFINE_integer("epoch", 16, "train epoch")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate of adam")
flags.DEFINE_float("beta2", 0.999, "beta2 of adam")
flags.DEFINE_string("data_dir", "./data", "path to dataset")
flags.DEFINE_string("out_dir", "./out", "directory for outputs")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "save checkpoints")
flags.DEFINE_boolean("train", True, "train or test")
FLAGS = flags.FLAGS


class MaxPropOptimizer(tf.train.Optimizer):
    def __init__(self, learning_rate=0.001, beta2=0.999, use_locking=False, name="MaxProp"):
        super(MaxPropOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta2 = beta2
        self._lr_t = None
        self._beta2_t = None

    def _prepare(self):
        self._lr_t = tf.convert_to_tensor(self._lr, name="learning_rate")
        self._beta2_t = tf.convert_to_tensor(self._beta2, name="beta2")

    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, "m", self._name)

    def _apply_dense(self, grad, var):
        lr_t = tf.cast(self._lr_t, var.dtype.base_dtype)
        beta2_t = tf.cast(self._beta2_t, var.dtype.base_dtype)
        if var.dtype.base_dtype == tf.float16:
            eps = 1e-7
        else:
            eps = 1e-8
        m = self.get_slot(var, "m")
        m_t = m.assign(tf.maximum(beta2_t * m + eps, tf.abs(grad)))
        g_t = grad / m_t
        var_update = tf.assign_sub(var, lr_t * g_t)
        return tf.group(*[var_update, m_t])

    def _apply_sparse(self, grad, var):
        return self._apply_dense(grad, var)


conv1d_index = 0


def conv1d_layer(input_tensor, size, dim, activation, scale, bias):
    global conv1d_index
    with tf.variable_scope('conv1d_' + str(conv1d_index)):
        W = tf.get_variable('W', (size, input_tensor.get_shape().as_list()[-1], dim),
                            dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=scale, maxval=scale))
        if bias:
            b = tf.get_variable('b', [dim], dtype=tf.float32, initializer=tf.constant_initializer(0))
        out = tf.nn.conv1d(input_tensor, W, stride=1, padding='SAME')
        if not bias:
            beta = tf.get_variable('beta', dim, dtype=tf.float32, initializer=tf.constant_initializer(0))
            gamma = tf.get_variable('gamma', dim, dtype=tf.float32, initializer=tf.constant_initializer(0))
            mean_running = tf.get_variable('mean', dim, dtype=tf.float32, initializer=tf.constant_initializer(0))
            variance_running = tf.get_variable('variance', dim, dtype=tf.float32, initializer=tf.constant_initializer(1))
            mean, variance = tf.nn.moments(out, axes=list(range(len(out.get_shape()) - 1)))

            def update_running_stat():
                decay = 0.99
                update_op = [mean_running.assign(mean_running * decay + mean * (1 - decay)),
                             variance_running.assign(variance_running * decay + variance * (1 - decay))]
                with tf.control_dependencies(update_op):
                    return tf.identity(mean), tf.identity(variance)

            m, v = tf.cond(tf.constant(False, dtype=tf.bool),
                           update_running_stat, lambda: (mean_running, variance_running))
            out = tf.nn.batch_normalization(out, m, v, beta, gamma, 1e-8)

        if activation == 'tanh':
            out = tf.nn.tanh(out)
        if activation == 'sigmoid':
            out = tf.nn.sigmoid(out)

        conv1d_index += 1
        return out


aconv1d_index = 0


def aconv1d_layer(input_tensor, size, rate, activation, scale, bias):
    global aconv1d_index
    with tf.variable_scope('aconv1d_' + str(aconv1d_index)):
        shape = input_tensor.get_shape().as_list()
        W = tf.get_variable('W', (1, size, shape[-1], shape[-1]), dtype=tf.float32,
                            initializer=tf.random_uniform_initializer(minval=-scale, maxval=scale))
        if bias:
            b = tf.get_variable('b', [shape[-1]], dtype=tf.float32, initializer=tf.constant_initializer(0))
        out = tf.nn.atrous_conv2d(tf.expand_dims(input_tensor, dim=1), W, rate=rate, padding='SAME')
        out = tf.squeeze(out, [1])

        if not bias:
            beta = tf.get_variable('beta', shape[-1], dtype=tf.float32, initializer=tf.constant_initializer(0))
            gamma = tf.get_variable('gamma', shape[-1], dtype=tf.float32, initializer=tf.constant_initializer(1))
            mean_running = tf.get_variable('mean', shape[-1], dtype=tf.float32, initializer=tf.constant_initializer(0))
            variance_running = tf.get_variable('variance', shape[-1], dtype=tf.float32, initializer=tf.constant_initializer(1))
            mean, variance = tf.nn.moments(out, axes=list(range(len(out.get_shape()) - 1)))

            def update_running_stat():
                decay = 0.99
                update_op = [mean_running.assign(mean_running * decay + mean * (1 - decay)),
                             variance_running.assign(variance_running * decay + variance * (1 - decay))]
                with tf.control_dependencies(update_op):
                    return tf.identity(mean), tf.identity(variance)

            m, v = tf.cond(tf.constant(False, dtype=tf.bool),
                           update_running_stat, lambda: (mean_running, variance_running))
            out = tf.nn.batch_normalization(out, m, v, beta, gamma, 1e-8)

            if activation == 'tanh':
                out = tf.nn.tanh(out)
            if activation == 'sigmoid':
                out = tf.nn.sigmoid(out)

            aconv1d_index += 1
            return out


def speech_to_text_network(n_dim=128, n_blocks=3):
    out = conv1d_layer(input_tensor=batch_input.X, size=1, dim=n_dim, activation='tanh', scale=0.14, bias=False)

    def residual_block(input_tensor, size, rate):
        conv_filter = aconv1d_layer(input_tensor, size=size, rate=rate, activation='tanh', scale=0.03, bias=False)
        conv_gate = aconv1d_layer(input_tensor, size=size, rate=rate, activation='sigmoid', scale=0.03, bias=False)
        _out = conv_filter * conv_gate
        _out = conv1d_layer(_out, size=1, dim=n_dim, activation='tanh', scale=0.08, bias=False)
        return _out + input_tensor, _out

    out, skip = residual_block(out, size=7, rate=1)
    for _ in range(n_blocks):
        for r in [2, 4, 8, 16]:
            out, s = residual_block(out, size=7, rate=r)
            skip += s

    logit = conv1d_layer(skip, size=1, dim=skip.get_shape().as_list()[-1], activation='tanh', scale=0.08, bias=False)
    logit = conv1d_layer(logit, size=1, dim=preprocess.words_size, activation=None, scale=0.04, bias=True)

    return logit


def train_speech_to_text_network():
    print("开始训练:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logit = speech_to_text_network()
    #epoch_now = tf.placeholder(shape=[], dtype=tf.float32, name='epoch_now')
    indices = tf.where(tf.not_equal(tf.cast(batch_input.Y, tf.float32), 0.))
    target = tf.SparseTensor(indices=indices, values=tf.gather_nd(batch_input.Y, indices),
                             dense_shape=tf.cast(tf.shape(batch_input.Y), tf.int64))
    loss = tf.nn.ctc_loss(target, logit, batch_input.sequence_len, time_major=False)
    #lr = FLAGS.learning_rate * (0.97 ** epoch_now)
    lr = tf.Variable(0.001, dtype=tf.float32, trainable=False)
    optimizer = MaxPropOptimizer(learning_rate=lr, beta2=FLAGS.beta2)
    var_list = [t for t in tf.trainable_variables()]
    gradient = optimizer.compute_gradients(loss, var_list=var_list)
    optimizer_op = optimizer.apply_gradients(gradient)
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        sess.run(init)

        saver = tf.train.Saver(tf.global_variables())

        for epoch in range(FLAGS.epoch):
            print(time.strftime('%Y-%m-%d %H:%M:%S'), time.localtime())
            print("第%d次循环迭代:" % epoch)
            #sess.run(lr, feed_dict={epoch_now: epoch})
            sess.run(tf.assign(lr,0.001 * (0.97 ** epoch)))


            global pointer
            pointer = 0
            for batch in range(batch_input.n_batch):
                batches_wavs, batches_labels = batch_input.get_next_batches(batch_input.batch_size)
                train_loss, _ = sess.run([loss, optimizer_op], feed_dict={batch_input.X: batches_wavs, batch_input.Y: batches_labels})
                print(epoch, batch, train_loss)
            if epoch % 5 == 0:
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                print("第%d次模型保存结果:" % (epoch//5))
                saver.save(sess, './model', global_step=epoch)
    print("结束训练时刻:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


if __name__ == '__main__':
    if FLAGS.train:
        train_speech_to_text_network()
    else:
        test.speech_to_text(preprocess.get_wav_files('d://data/wav/test'))

