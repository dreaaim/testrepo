import os
import numpy as np
import tensorflow as tf
import time

flags = tf.flags
flags.DEFINE_integer("epoch", 16, "train epoch")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate of adam")
flags.DEFINE_float("beta2", 0.999, "beta2 of adam")
flags.DEFINE_string("data_dir", "./data", "path to dataset")
flags.DEFINE_string("out_dir", "./out", "directory for outputs")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "save checkpoints")
flags.DEFINE_string("train", True, "train or test")
FLAGS = flags.FLAGS


class MaxPropOptimizer(tf.train.Optimizer):
    def __init__(self, learning_rate = 0.001, beta2 = 0.999, use_locking = False, name = "MaxProp"):
        super(MaxPropOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta2 = beta2
        self._lr_t = None
        self._beta2_t = None

    def _prepare(self):
        self._lr_t = tf.convert_to_tensor(self._lr, name = "learning_rate")
        self._beta2_t = tf.convert_to_tensor(self._beta2, name = "beta2")

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
        m_t = m.assign(tf.maximum(beta2_t * m + eps. tf.abs(grad)))
        g_t = grad / m_t
        var_update = tf.assign_sub(var, lr_t * g_t)
        return tf.group(*[var_update, m_t])

    def _apply_sparse(self, grad, var):
        return self._apply_dense(grad, var)


def train_speech_to_text_network():
    print("开始训练:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logit = speech_to_text_network()

