import tarfile
import os
from mxnet import nd
from mxnet.gluon import utils as gutils, data as gdata
from .base import Vocab

__all__ = ['load_data_imdb']

def load_data_imdb(batch_size, max_len=500):
    """Download an IMDB dataset, return the vocabulary and iterators."""

    data_dir = './'
    url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    fname = gutils.download(url, data_dir)
    with tarfile.open(fname, 'r') as f:
        f.extractall(data_dir)

    def read_imdb(folder='train'):
        data, labels = [], []
        for label in ['pos', 'neg']:
            folder_name = os.path.join(data_dir, 'aclImdb', folder, label)
            for file in os.listdir(folder_name):
                with open(os.path.join(folder_name, file), 'rb') as f:
                    review = f.read().decode('utf-8').replace('\n', '')
                    data.append(review)
                    labels.append(1 if label == 'pos' else 0)
        return data, labels

    train_data, test_data = read_imdb('train'), read_imdb('test')

    def tokenize(sentences):
        return [line.split(' ') for line in sentences]

    train_tokens = tokenize(train_data[0])
    test_tokens = tokenize(test_data[0])

    vocab = Vocab([tk for line in train_tokens for tk in line], min_freq=5)

    def pad(x):
        return x[:max_len] if len(x) > max_len else x + [vocab.unk] * (max_len - len(x))

    train_features = nd.array([pad(vocab[line]) for line in train_tokens])
    test_features = nd.array([pad(vocab[line]) for line in test_tokens])

    train_set = gdata.ArrayDataset(train_features, train_data[1])
    test_set = gdata.ArrayDataset(test_features, test_data[1])
    train_iter = gdata.DataLoader(train_set, batch_size, shuffle=True)
    test_iter = gdata.DataLoader(test_set, batch_size)

    return vocab, train_iter, test_iter
