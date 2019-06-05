import tensorflow as tf
import preprocess
from ops import *

def speech_to_text_network(n_dim=128, n_blocks=3):
    out = conv1d_layer(input_tensor=X, size=1, dim=n_dim, activation='tanh', scale=0.14, bias=False)

    def residual_block(input_sensor, size, rate):
        conv_filter = aconv1d_layer(input_tensor, size=size, rate=rate, activation='tanh', scale=0.03, bias=False)
        conv_gate = aconv1d_layer(input_tensor, size=size, rate=rate, activation='sigmoid', scale=0.03, bias=False)
        out = conv_filter * conv_gate
        out = conv1d_layer(out, size=1, dim=n_dim, activation='tanh', scale=0.08, bias=False)
        return out + input_sensor, out

    skip = 0
    for _ in range(n_blocks):
        for r in [1, 2, 4, 8, 16]:
            out, s = residual_block(out, size=7, rate=r)
            skip += s

    logit = conv1d_layer(skip, size=1, dim=skip.get_shape().as_list()[-1], activation='tanh', scale=0.08, bias=False)
    logit = conv1d_layer(logit, size=1, dim=preprocess.words_size, activation=None, scale=0.04, bias=True)

    return logit