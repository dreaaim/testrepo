import tensorflow as tf
import preprocess
import batch_input
from ops import *


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