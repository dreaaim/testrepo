import tensorflow as tf
import numpy as np
import os
from collections import Counter
import librosa
import time
import preprocess

batch_size = 16
n_batch = len(preprocess.wav_files)//batch_size
pointer = 0


def get_next_batches(_batch_size=batch_size):
    global pointer
    batches_wavs = []
    batches_labels = []
    for i in range(_batch_size):
        wav, sr = librosa.load(preprocess.wav_files[pointer], mono=True)
        _mfcc = np.transpose(librosa.feature.mfcc(wav, sr), [1, 0])
        batches_wavs.append(_mfcc.tolist())
        batches_labels.append(preprocess.labels_vector[pointer])
        pointer += 1

    for mfcc_wavs in batches_wavs:
        while len(mfcc_wavs.__str__()) < preprocess.wav_max_len:
            mfcc_wavs.append([0] * 20)
    for label in batches_labels:
        while len(label) < preprocess.label_max_len:
            label.append(0)
    return batches_wavs, batches_labels


X = tf.placeholder(dtype=tf.float32, shape=[batch_size, None, 20])
sequence_len = tf.reduce_sum(tf.cast(tf.not_equal(tf.reduce_sum(X, reduction_indices=2), 0.), tf.int32),
                             reduction_indices=1)
Y = tf.placeholder(dtype=tf.int32, shape=[batch_size, None])

