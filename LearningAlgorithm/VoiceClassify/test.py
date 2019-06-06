import tensorflow as tf
import numpy as np
import librosa
import layers
import batch_input
import preprocess


def speech_to_text(wav_file):
    wav, sr = librosa.load(wav_file, mono=True)
    mfcc = np.transpose(np.expand_dims(librosa.feature.mfcc(wav, sr), axis=0), [0, 2, 1])
    logit = layers.speech_to_text_network()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))

        decoded = tf.transpose(logit, perm=[1, 0, 2])
        decoded, _ = tf.nn.ctc_beam_search_decoder(decoded, batch_input.sequence_len, merge_repeated=False)
        predict = tf.sparse_to_dense(decoded[0].indices, decoded[0].shape, decoded[0].values) + 1
        output = sess.run(decoded, feed_dict={X : mfcc})
        print(output)


if __name__ == "main":
    speech_to_text(preprocess.wav_files)
