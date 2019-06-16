"""The train module contains functions for neural network training"""
import numpy as np
import math
import time

import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn, utils as gutils
from ..d2l import try_gpu, linreg
from ..figure import set_figsize, plt


def predict_sentiment(net, vocab, sentence):
    """Predict the sentiment of a given sentence."""
    sentence = nd.array(vocab[sentence.split()], ctx=try_gpu())
    label = nd.argmax(net(sentence.reshape((1, -1))), axis=1)
    return 'positive' if label.asscalar() == 1 else 'negative'
