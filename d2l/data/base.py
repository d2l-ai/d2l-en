import os
import random
import numpy as np
from mxnet import nd
from mxnet.gluon import utils as gutils


def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    """Sample mini-batches in a consecutive order from sequential data."""
    # Offset for the iterator over the data for uniform starts
    offset = int(random.uniform(0, num_steps))
    # Slice out data - ignore num_steps and just wrap around
    num_indices = ((len(corpus_indices) - offset) // batch_size) * batch_size
    indices = nd.array(corpus_indices[offset:(offset + num_indices)], ctx=ctx)
    indices = indices.reshape((batch_size, -1))
    # Need to leave one last token since targets are shifted by 1
    num_epochs = ((num_indices // batch_size) - 1) // num_steps

    for i in range(0, num_epochs * num_steps, num_steps):
        X = indices[:, i:(i+num_steps)]
        Y = indices[:, (i+1):(i+1+num_steps)]
        yield X, Y


def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    """Sample mini-batches in a random order from sequential data."""
    # Offset for the iterator over the data
    offset = int(random.uniform(0, num_steps))
    # Subtract 1 extra since we need to account for the sequence length
    num_examples = ((len(corpus_indices) - offset - 1) // num_steps) - 1
    # Discard half empty batches
    num_batches = num_examples // batch_size
    example_indices = list(
        range(offset, offset + num_examples * num_steps, num_steps))
    random.shuffle(example_indices)

    # This returns a sequence of the length num_steps starting from pos.
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]

    for i in range(0, batch_size * num_batches, batch_size):
        # batch_size indicates the random examples read each time.
        batch_indices = example_indices[i:(i+batch_size)]
        X = [_data(j) for j in batch_indices]
        Y = [_data(j + 1) for j in batch_indices]
        yield nd.array(X, ctx), nd.array(Y, ctx)



def get_data_ch7():
    """Get the data set used in Chapter 7."""
    data = np.genfromtxt('../data/airfoil_self_noise.dat', delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return nd.array(data[:, :-1]), nd.array(data[:, -1])

def load_data_time_machine():
    """Load the time machine data set (available in the English book)."""
    with open('../data/timemachine.txt') as f:
        corpus_chars = f.read()
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ').lower()
    corpus_chars = corpus_chars[0:10000]
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size



def mkdir_if_not_exist(path):
    """Make a directory if it does not exist."""
    if not os.path.exists(os.path.join(*path)):
        os.makedirs(os.path.join(*path))
