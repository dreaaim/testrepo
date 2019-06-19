import os
import numpy as np
import tensorflow as tf
import time
import layers
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


def train_speech_to_text_network():
    print("开始训练:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logit = layers.speech_to_text_network()

    indices = tf.where(tf.not_equal(tf.cast(batch_input.Y, tf.float32), 0.))
    target = tf.SparseTensor(indices=indices, values=tf.gather_nd(batch_input.Y, indices),
                             dense_shape=tf.cast(tf.shape(batch_input.Y), tf.int64))
    loss = tf.nn.ctc_loss(target, logit, batch_input.sequence_len, time_major=False)
    lr = FLAGS.learning_rate
    optimizer = MaxPropOptimizer(learning_rate=lr, beta2=FLAGS.beta2)
    var_list = [t for t in tf.trainable_variables()]
    gradient = optimizer.compute_gradients(loss, var_list=var_list)
    optimizer_op = optimizer.apply_gradients(gradient)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables())

        for epoch in range(FLAGS.epoch):
            print(time.strftime('%Y-%m-%d %H:%M:%S'), time.localtime())
            print("第%d次循环迭代:" % epoch)
            sess.run(tf.assign(lr, 0.001 * (0.97 ** epoch)))

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

