import os
import random
import numpy as np
import zipfile
import collections
from mxnet import nd
from mxnet.gluon import utils as gutils, data as gdata

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

class Vocab(object):
    def __init__(self, tokens, min_freq):
        counter = collections.Counter(tokens)
        token_freqs = sorted(counter.items(), key=lambda x: x[0])
        token_freqs.sort(key=lambda x: x[1], reverse=True)
        self.pad, self.bos, self.eos, self.unk = (0, 1, 2, 3)
        tokens = ['<pad>', '<bos>', '<eos>', '<unk>'] + [
            token for token, freq in token_freqs if freq >= min_freq]
        self.idx_to_token = []
        self.token_to_idx = dict()
        for token in tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        else:
            return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        else:
            return [self.idx_to_token[index] for index in indices]

def load_data_nmt(batch_size, max_len, num_examples=1000):
    # download and preprocess
    def preprocess_raw(text):
        text = text.replace('\u202f', ' ').replace('\xa0', ' ')
        out = ''
        for i, char in enumerate(text.lower()):
            if char in (',', '!', '.') and text[i-1] != ' ':
                out += ' '
            out += char
        return out
    fname = gutils.download('http://www.manythings.org/anki/fra-eng.zip')
    with zipfile.ZipFile(fname, 'r') as f:
        raw_text = f.read('fra.txt').decode("utf-8")
    text = preprocess_raw(raw_text)

    # tokenize
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if i >= num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))

    # build vocab
    def build_vocab(tokens):
        tokens = [token for line in tokens for token in line]
        return Vocab(tokens, min_freq=2)
    src_vocab, tgt_vocab = build_vocab(source), build_vocab(target)

    # convert to index arrays
    def pad(line, max_len, padding_token):
        if len(line) > max_len:
            return line[:max_len]
        return line + [padding_token] * (max_len - len(line))

    def build_array(lines, vocab, max_len, is_source):
        lines = [vocab[line] for line in lines]
        if not is_source:
            lines = [[vocab.bos] + line + [vocab.eos] for line in lines]
        array = nd.array([pad(line, max_len, vocab.pad) for line in lines])
        valid_len = (array != vocab.pad).sum(axis=1)
        return array, valid_len

    src_array, src_valid_len = build_array(source, src_vocab, max_len, True)
    tgt_array, tgt_valid_len = build_array(target, tgt_vocab, max_len, False)

    # construct data iterator
    train_set = gdata.ArrayDataset(src_array, src_valid_len, tgt_array, tgt_valid_len)
    train_iter = gdata.DataLoader(train_set, batch_size, shuffle=True)

    return src_vocab, tgt_vocab, train_iter
