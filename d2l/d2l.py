# This file is generated automatically through:
#    d2lbook build lib
# Don't edit it directly

import sys
d2l = sys.modules[__name__]

# Defined in file: ./chapter_preface/index.md
import collections
from collections import defaultdict
from IPython import display
import math
from matplotlib import pyplot as plt
from mxnet import autograd, context, gluon, image, init, np, npx
from mxnet.gluon import nn, rnn
import os
import pandas as pd
import random
import re
import shutil
import sys
import tarfile
import time
import zipfile


# Defined in file: ./chapter_preliminaries/pandas.md
def mkdir_if_not_exist(path):
    if not isinstance(path, str):
        path = os.path.join(*path)
    if not os.path.exists(path):
        os.makedirs(path)


# Defined in file: ./chapter_preliminaries/calculus.md
def use_svg_display():
    """Use the svg format to display a plot in Jupyter."""
    display.set_matplotlib_formats('svg')


# Defined in file: ./chapter_preliminaries/calculus.md
def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size for matplotlib."""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize


# Defined in file: ./chapter_preliminaries/calculus.md
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib."""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


# Defined in file: ./chapter_preliminaries/calculus.md
def plot(X, Y=None, xlabel=None, ylabel=None, legend=[], xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=['-', 'm--', 'g-.', 'r:'], figsize=(3.5, 2.5), axes=None):
    """Plot data points."""
    d2l.set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()

    # Return True if X (ndarray or list) has 1 axis
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


# Defined in file: ./chapter_linear-networks/linear-regression.md
class Timer(object):
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        # Start the timer
        self.start_time = time.time()

    def stop(self):
        # Stop the timer and record the time in a list
        self.times.append(time.time() - self.start_time)
        return self.times[-1]

    def avg(self):
        # Return the average time
        return sum(self.times)/len(self.times)

    def sum(self):
        # Return the sum of time
        return sum(self.times)

    def cumsum(self):
        # Return the accumuated times
        return np.array(self.times).cumsum().tolist()


# Defined in file: ./chapter_linear-networks/linear-regression-scratch.md
def synthetic_data(w, b, num_examples):
    """Generate y = X w + b + noise."""
    X = np.random.normal(0, 1, (num_examples, len(w)))
    y = np.dot(X, w) + b
    y += np.random.normal(0, 0.01, y.shape)
    return X, y


# Defined in file: ./chapter_linear-networks/linear-regression-scratch.md
def linreg(X, w, b):
    return np.dot(X, w) + b


# Defined in file: ./chapter_linear-networks/linear-regression-scratch.md
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# Defined in file: ./chapter_linear-networks/linear-regression-scratch.md
def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size


# Defined in file: ./chapter_linear-networks/linear-regression-gluon.md
def load_array(data_arrays, batch_size, is_train=True):
    """Construct a Gluon data loader"""
    dataset = gluon.data.ArrayDataset(*data_arrays)
    return gluon.data.DataLoader(dataset, batch_size, shuffle=is_train)


# Defined in file: ./chapter_linear-networks/fashion-mnist.md
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


# Defined in file: ./chapter_linear-networks/fashion-mnist.md
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img.asnumpy())
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


# Defined in file: ./chapter_linear-networks/fashion-mnist.md
def get_dataloader_workers(num_workers=4):
    # 0 means no additional process is used to speed up the reading of data.
    if sys.platform.startswith('win'):
        return 0
    else:
        return num_workers


# Defined in file: ./chapter_linear-networks/fashion-mnist.md
def load_data_fashion_mnist(batch_size, resize=None):
    """Download the Fashion-MNIST dataset and then load into memory."""
    dataset = gluon.data.vision
    trans = [dataset.transforms.Resize(resize)] if resize else []
    trans.append(dataset.transforms.ToTensor())
    trans = dataset.transforms.Compose(trans)
    mnist_train = dataset.FashionMNIST(train=True).transform_first(trans)
    mnist_test = dataset.FashionMNIST(train=False).transform_first(trans)
    return (gluon.data.DataLoader(mnist_train, batch_size, shuffle=True,
                                  num_workers=get_dataloader_workers()),
            gluon.data.DataLoader(mnist_test, batch_size, shuffle=False,
                                  num_workers=get_dataloader_workers()))


# Defined in file: ./chapter_linear-networks/softmax-regression-scratch.md
def accuracy(y_hat, y):
    if y_hat.shape[1] > 1:
        return float((y_hat.argmax(axis=1) == y.astype('float32')).sum())
    else:
        return float((y_hat.astype('int32') == y.astype('int32')).sum())


# Defined in file: ./chapter_linear-networks/softmax-regression-scratch.md
def evaluate_accuracy(net, data_iter):
    metric = Accumulator(2)  # num_corrected_examples, num_examples
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.size)
    return metric[0] / metric[1]


# Defined in file: ./chapter_linear-networks/softmax-regression-scratch.md
class Accumulator(object):
    """Sum a list of numbers over time."""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a+float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Defined in file: ./chapter_linear-networks/softmax-regression-scratch.md
def train_epoch_ch3(net, train_iter, loss, updater):
    metric = Accumulator(3)  # train_loss_sum, train_acc_sum, num_examples
    if isinstance(updater, gluon.Trainer):
        updater = updater.step
    for X, y in train_iter:
        # Compute gradients and update parameters
        with autograd.record():
            y_hat = net(X)
            l = loss(y_hat, y)
        l.backward()
        updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.size)
    # Return training loss and training accuracy
    return metric[0]/metric[2], metric[1]/metric[2]


# Defined in file: ./chapter_linear-networks/softmax-regression-scratch.md
class Animator(object):
    def __init__(self, xlabel=None, ylabel=None, legend=[], xlim=None,
                 ylim=None, xscale='linear', yscale='linear', fmts=None,
                 nrows=1, ncols=1, figsize=(3.5, 2.5)):
        """Incrementally plot multiple lines."""
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda to capture arguments
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        """Add multiple data points into the figure."""
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        if not self.fmts:
            self.fmts = ['-'] * n
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


# Defined in file: ./chapter_linear-networks/softmax-regression-scratch.md
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                        ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch+1, train_metrics+(test_acc,))


# Defined in file: ./chapter_linear-networks/softmax-regression-scratch.md
def predict_ch3(net, test_iter, n=6):
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true+'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(X[0:n].reshape(n, 28, 28), 1, n, titles=titles[0:n])


# Defined in file: ./chapter_multilayer-perceptrons/underfit-overfit.md
def evaluate_loss(net, data_iter, loss):
    """Evaluate the loss of a model on the given dataset."""
    metric = d2l.Accumulator(2)  # sum_loss, num_examples
    for X, y in data_iter:
        metric.add(loss(net(X), y).sum(), y.size)
    return metric[0] / metric[1]


# Defined in file: ./chapter_multilayer-perceptrons/kaggle-house-price.md
DATA_HUB = dict()


# Defined in file: ./chapter_multilayer-perceptrons/kaggle-house-price.md
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'


# Defined in file: ./chapter_multilayer-perceptrons/kaggle-house-price.md
def download(name, cache_dir='../data'):
    """Download a file inserted into DATA_HUB, return the local filename."""
    assert name in DATA_HUB, "%s doesn't exist" % name
    url, sha1 = DATA_HUB[name]
    d2l.mkdir_if_not_exist(cache_dir)
    return gluon.utils.download(url, cache_dir, sha1_hash=sha1)


# Defined in file: ./chapter_multilayer-perceptrons/kaggle-house-price.md
def download_extract(name, folder=None):
    """Download and extract a zip/tar file."""
    fname = download(name)
    base_dir = os.path.dirname(fname) 
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext == '.tar' or ext == '.gz':
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'Only zip/tar files can be extracted'
    fp.extractall(base_dir)
    if folder:
        return base_dir + '/' + folder + '/'
    else:
        return data_dir + '/'


# Defined in file: ./chapter_multilayer-perceptrons/kaggle-house-price.md
def download_all():
    """Download all files in the DATA_HUB"""
    for name in DATA_HUB:
        download(name)


# Defined in file: ./chapter_multilayer-perceptrons/kaggle-house-price.md
DATA_HUB['kaggle_house_train'] = (
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')


# Defined in file: ./chapter_multilayer-perceptrons/kaggle-house-price.md
DATA_HUB['kaggle_house_test'] = (
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')


# Defined in file: ./chapter_deep-learning-computation/use-gpu.md
def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    return npx.gpu(i) if npx.num_gpus() >= i + 1 else npx.cpu()


# Defined in file: ./chapter_deep-learning-computation/use-gpu.md
def try_all_gpus():
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    ctxes = [npx.gpu(i) for i in range(npx.num_gpus())]
    return ctxes if ctxes else [npx.cpu()]


# Defined in file: ./chapter_convolutional-neural-networks/conv-layer.md
def corr2d(X, K):
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = np.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y


# Defined in file: ./chapter_convolutional-neural-networks/lenet.md
def evaluate_accuracy_gpu(net, data_iter, ctx=None):
    if not ctx:  # Query the first device the first parameter is on
        ctx = list(net.collect_params().values())[0].list_ctx()[0]
    metric = d2l.Accumulator(2)  # num_corrected_examples, num_examples
    for X, y in data_iter:
        X, y = X.as_in_context(ctx), y.as_in_context(ctx)
        metric.add(d2l.accuracy(net(X), y), y.size)
    return metric[0]/metric[1]


# Defined in file: ./chapter_convolutional-neural-networks/lenet.md
def train_ch6(net, train_iter, test_iter, num_epochs, lr, ctx=d2l.try_gpu()):
    net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(),
                            'sgd', {'learning_rate': lr})
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # train_loss, train_acc, num_examples
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            # Here is the only difference compared to train_epoch_ch3
            X, y = X.as_in_context(ctx), y.as_in_context(ctx)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(X.shape[0])
            metric.add(l.sum(), d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_loss, train_acc = metric[0]/metric[2], metric[1]/metric[2]
            if (i+1) % 50 == 0:
                animator.add(epoch + i/len(train_iter),
                             (train_loss, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch+1, (None, None, test_acc))
    print('loss %.3f, train acc %.3f, test acc %.3f' % (
        train_loss, train_acc, test_acc))
    print('%.1f examples/sec on %s' % (metric[2]*num_epochs/timer.sum(), ctx))


# Defined in file: ./chapter_convolutional-modern/resnet.md
class Residual(nn.Block):
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1,
                               strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1,
                                   strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def forward(self, X):
        Y = npx.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return npx.relu(Y + X)


# Defined in file: ./chapter_recurrent-neural-networks/text-preprocessing.md
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')


# Defined in file: ./chapter_recurrent-neural-networks/text-preprocessing.md
def read_time_machine():
    """Load the time machine book into a list of sentences."""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line.strip().lower())
            for line in lines]


# Defined in file: ./chapter_recurrent-neural-networks/text-preprocessing.md
def tokenize(lines, token='word'):
    """Split sentences into word or char tokens."""
    if token == 'word':
        return [line.split(' ') for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('ERROR: unknown token type '+token)


# Defined in file: ./chapter_recurrent-neural-networks/text-preprocessing.md
class Vocab(object):
    def __init__(self, tokens, min_freq=0, reserved_tokens=[]):
        # Sort according to frequencies
        counter = count_corpus(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[0])
        self.token_freqs.sort(key=lambda x: x[1], reverse=True)
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens
        uniq_tokens += [token for token, freq in self.token_freqs
                        if freq >= min_freq and token not in uniq_tokens]
        self.idx_to_token, self.token_to_idx = [], dict()
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]


# Defined in file: ./chapter_recurrent-neural-networks/text-preprocessing.md
def count_corpus(sentences):
    # Flatten a list of token lists into a list of tokens
    tokens = [tk for line in sentences for tk in line]
    return collections.Counter(tokens)


# Defined in file: ./chapter_recurrent-neural-networks/text-preprocessing.md
def load_corpus_time_machine(max_tokens=-1):
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    corpus = [vocab[tk] for line in tokens for tk in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


# Defined in file: ./chapter_recurrent-neural-networks/lang-model.md
def seq_data_iter_random(corpus, batch_size, num_steps):
    # Offset the iterator over the data for uniform starts
    corpus = corpus[random.randint(0, num_steps):]
    # Subtract 1 extra since we need to account for label
    num_examples = ((len(corpus) - 1) // num_steps)
    example_indices = list(range(0, num_examples * num_steps, num_steps))
    random.shuffle(example_indices)

    def data(pos):
        # This returns a sequence of the length num_steps starting from pos
        return corpus[pos: pos + num_steps]

    # Discard half empty batches
    num_batches = num_examples // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # Batch_size indicates the random examples read each time
        batch_indices = example_indices[i:(i+batch_size)]
        X = [data(j) for j in batch_indices]
        Y = [data(j + 1) for j in batch_indices]
        yield np.array(X), np.array(Y)


# Defined in file: ./chapter_recurrent-neural-networks/lang-model.md
def seq_data_iter_consecutive(corpus, batch_size, num_steps):
    # Offset for the iterator over the data for uniform starts
    offset = random.randint(0, num_steps)
    # Slice out data - ignore num_steps and just wrap around
    num_indices = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = np.array(corpus[offset:offset+num_indices])
    Ys = np.array(corpus[offset+1:offset+1+num_indices])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_batches * num_steps, num_steps):
        X = Xs[:, i:(i+num_steps)]
        Y = Ys[:, i:(i+num_steps)]
        yield X, Y


# Defined in file: ./chapter_recurrent-neural-networks/lang-model.md
class SeqDataLoader(object):
    """A iterator to load sequence data."""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_consecutive
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


# Defined in file: ./chapter_recurrent-neural-networks/lang-model.md
def load_data_time_machine(batch_size, num_steps, use_random_iter=False,
                           max_tokens=10000):
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


# Defined in file: ./chapter_recurrent-neural-networks/rnn-scratch.md
class RNNModelScratch(object):
    """A RNN Model based on scratch implementations."""

    def __init__(self, vocab_size, num_hiddens, ctx,
                 get_params, init_state, forward):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, ctx)
        self.init_state, self.forward_fn = init_state, forward

    def __call__(self, X, state):
        X = npx.one_hot(X.T, self.vocab_size)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, ctx):
        return self.init_state(batch_size, self.num_hiddens, ctx)


# Defined in file: ./chapter_recurrent-neural-networks/rnn-scratch.md
def predict_ch8(prefix, num_predicts, model, vocab, ctx):
    state = model.begin_state(batch_size=1, ctx=ctx)
    outputs = [vocab[prefix[0]]]

    def get_input():
        return np.array([outputs[-1]], ctx=ctx).reshape(1, 1)
    for y in prefix[1:]:  # Warmup state with prefix
        _, state = model(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_predicts):  # Predict num_predicts steps
        Y, state = model(get_input(), state)
        outputs.append(int(Y.argmax(axis=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


# Defined in file: ./chapter_recurrent-neural-networks/rnn-scratch.md
def grad_clipping(model, theta):
    if isinstance(model, gluon.Block):
        params = [p.data() for p in model.collect_params().values()]
    else:
        params = model.params
    norm = math.sqrt(sum((p.grad ** 2).sum() for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


# Defined in file: ./chapter_recurrent-neural-networks/rnn-scratch.md
def train_epoch_ch8(model, train_iter, loss, updater, ctx, use_random_iter):
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # loss_sum, num_examples
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # Initialize state when either it is the first iteration or
            # using random sampling.
            state = model.begin_state(batch_size=X.shape[0], ctx=ctx)
        else:
            for s in state:
                s.detach()
        y = Y.T.reshape(-1)
        X, y = X.as_in_context(ctx), y.as_in_context(ctx)
        with autograd.record():
            py, state = model(X, state)
            l = loss(py, y).mean()
        l.backward()
        grad_clipping(model, 1)
        updater(batch_size=1)  # Since used mean already
        metric.add(l * y.size, y.size)
    return math.exp(metric[0]/metric[1]), metric[1]/timer.stop()


# Defined in file: ./chapter_recurrent-neural-networks/rnn-scratch.md
def train_ch8(model, train_iter, vocab, lr, num_epochs, ctx,
              use_random_iter=False):
    # Initialize
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[1, num_epochs])
    if isinstance(model, gluon.Block):
        model.initialize(ctx=ctx, force_reinit=True, init=init.Normal(0.01))
        trainer = gluon.Trainer(model.collect_params(),
                                'sgd', {'learning_rate': lr})

        def updater(batch_size):
            return trainer.step(batch_size)
    else:
        def updater(batch_size):
            return d2l.sgd(model.params, lr, batch_size)

    def predict(prefix):
        return predict_ch8(prefix, 50, model, vocab, ctx)

    # Train and check the progress.
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            model, train_iter, loss, updater, ctx, use_random_iter)
        if epoch % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch+1, [ppl])
    print('Perplexity %.1f, %d tokens/sec on %s' % (ppl, speed, ctx))
    print(predict('time traveller'))
    print(predict('traveller'))


# Defined in file: ./chapter_recurrent-neural-networks/rnn-gluon.md
class RNNModel(nn.Block):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = nn.Dense(vocab_size)

    def forward(self, inputs, state):
        X = npx.one_hot(inputs.T, self.vocab_size)
        Y, state = self.rnn(X, state)
        # The fully connected layer will first change the shape of Y to
        # (num_steps * batch_size, num_hiddens). Its output shape is
        # (num_steps * batch_size, vocab_size).
        output = self.dense(Y.reshape(-1, Y.shape[-1]))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)


# Defined in file: ./chapter_recurrent-modern/machine-translation.md
d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')


# Defined in file: ./chapter_recurrent-modern/machine-translation.md
def read_data_nmt():
    data_dir = d2l.download_extract('fra-eng')
    with open(data_dir + 'fra.txt', 'r') as f:
        return f.read()


# Defined in file: ./chapter_recurrent-modern/machine-translation.md
def preprocess_nmt(text):
    def no_space(char, prev_char):
        return char in set(',.!') and prev_char != ' '

    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    out = [' ' + char if i > 0 and no_space(char, text[i-1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)


# Defined in file: ./chapter_recurrent-modern/machine-translation.md
def tokenize_nmt(text, num_examples=None):
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


# Defined in file: ./chapter_recurrent-modern/machine-translation.md
def trim_pad(line, num_steps, padding_token):
    if len(line) > num_steps:
        return line[:num_steps]  # Trim
    return line + [padding_token] * (num_steps - len(line))  # Pad


# Defined in file: ./chapter_recurrent-modern/machine-translation.md
def build_array(lines, vocab, num_steps, is_source):
    lines = [vocab[l] for l in lines]
    if not is_source:
        lines = [[vocab['<bos>']] + l + [vocab['<eos>']] for l in lines]
    array = np.array([trim_pad(l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).sum(axis=1)
    return array, valid_len


# Defined in file: ./chapter_recurrent-modern/machine-translation.md
def load_data_nmt(batch_size, num_steps, num_examples=1000):
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = d2l.Vocab(source, min_freq=3, 
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=3, 
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array(
        source, src_vocab, num_steps, True)
    tgt_array, tgt_valid_len = build_array(
        target, tgt_vocab, num_steps, False)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return src_vocab, tgt_vocab, data_iter


# Defined in file: ./chapter_recurrent-modern/encoder-decoder.md
class Encoder(nn.Block):
    """The base encoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError


# Defined in file: ./chapter_recurrent-modern/encoder-decoder.md
class Decoder(nn.Block):
    """The base decoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


# Defined in file: ./chapter_recurrent-modern/encoder-decoder.md
class EncoderDecoder(nn.Block):
    """The base class for the encoder-decoder architecture."""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)


# Defined in file: ./chapter_recurrent-modern/seq2seq.md
class Seq2SeqEncoder(d2l.Encoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.LSTM(num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        X = self.embedding(X)  # X shape: (batch_size, seq_len, embed_size)
        # RNN needs first axes to be timestep, i.e., seq_len
        X = X.swapaxes(0, 1)
        state = self.rnn.begin_state(batch_size=X.shape[1], ctx=X.context)
        out, state = self.rnn(X, state)
        # out shape: (seq_len, batch_size, num_hiddens)
        # state shape: (num_layers, batch_size, num_hiddens),
        # where "state" contains the hidden state and the memory cell
        return out, state


# Defined in file: ./chapter_recurrent-modern/seq2seq.md
class Seq2SeqDecoder(d2l.Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.LSTM(num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Dense(vocab_size, flatten=False)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        X = self.embedding(X).swapaxes(0, 1)
        out, state = self.rnn(X, state)
        # Make the batch to be the first dimension to simplify loss
        # computation
        out = self.dense(out).swapaxes(0, 1)
        return out, state


# Defined in file: ./chapter_recurrent-modern/seq2seq.md
class MaskedSoftmaxCELoss(gluon.loss.SoftmaxCELoss):
    # pred shape: (batch_size, seq_len, vocab_size)
    # label shape: (batch_size, seq_len)
    # valid_len shape: (batch_size, )
    def forward(self, pred, label, valid_len):
        # weights shape: (batch_size, seq_len, 1)
        weights = np.expand_dims(np.ones_like(label), axis=-1)
        weights = npx.sequence_mask(weights, valid_len, True, axis=1)
        return super(MaskedSoftmaxCELoss, self).forward(pred, label, weights)


# Defined in file: ./chapter_recurrent-modern/seq2seq.md
def train_s2s_ch9(model, data_iter, lr, num_epochs, ctx):
    model.initialize(init.Xavier(), force_reinit=True, ctx=ctx)
    trainer = gluon.Trainer(model.collect_params(),
                            'adam', {'learning_rate': lr})
    loss = MaskedSoftmaxCELoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs], ylim=[0, 0.25])
    for epoch in range(1, num_epochs + 1):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # loss_sum, num_tokens
        for batch in data_iter:
            X, X_vlen, Y, Y_vlen = [x.as_in_context(ctx) for x in batch]
            Y_input, Y_label, Y_vlen = Y[:, :-1], Y[:, 1:], Y_vlen-1
            with autograd.record():
                Y_hat, _ = model(X, Y_input, X_vlen, Y_vlen)
                l = loss(Y_hat, Y_label, Y_vlen)
            l.backward()
            d2l.grad_clipping(model, 1)
            num_tokens = Y_vlen.sum()
            trainer.step(num_tokens)
            metric.add(l.sum(), num_tokens)
        if epoch % 10 == 0:
            animator.add(epoch, (metric[0]/metric[1],))
    print('loss %.3f, %d tokens/sec on %s ' % (
        metric[0]/metric[1], metric[1]/timer.stop(), ctx))


# Defined in file: ./chapter_recurrent-modern/seq2seq.md
def predict_s2s_ch9(model, src_sentence, src_vocab, tgt_vocab, num_steps,
                    ctx):
    src_tokens = src_vocab[src_sentence.lower().split(' ')]
    enc_valid_len = np.array([len(src_tokens)], ctx=ctx)
    src_tokens = d2l.trim_pad(src_tokens, num_steps, src_vocab['<pad>'])
    enc_X = np.array(src_tokens, ctx=ctx)
    # Add the batch_size dimension
    enc_outputs = model.encoder(np.expand_dims(enc_X, axis=0),
                                enc_valid_len)
    dec_state = model.decoder.init_state(enc_outputs, enc_valid_len)
    dec_X = np.expand_dims(np.array([tgt_vocab['<bos>']], ctx=ctx), axis=0)
    predict_tokens = []
    for _ in range(num_steps):
        Y, dec_state = model.decoder(dec_X, dec_state)
        # The token with highest score is used as the next timestep input
        dec_X = Y.argmax(axis=2)
        py = dec_X.squeeze(axis=0).astype('int32').item()
        if py == tgt_vocab['<eos>']:
            break
        predict_tokens.append(py)
    return ' '.join(tgt_vocab.to_tokens(predict_tokens))


# Defined in file: ./chapter_attention-mechanisms/attention.md
def masked_softmax(X, valid_len):
    # X: 3-D tensor, valid_len: 1-D or 2-D tensor
    if valid_len is None:
        return npx.softmax(X)
    else:
        shape = X.shape
        if valid_len.ndim == 1:
            valid_len = valid_len.repeat(shape[1], axis=0)
        else:
            valid_len = valid_len.reshape(-1)
        # Fill masked elements with a large negative, whose exp is 0
        X = npx.sequence_mask(X.reshape(-1, shape[-1]), valid_len, True,
                              axis=1, value=-1e6)
        return npx.softmax(X).reshape(shape)


# Defined in file: ./chapter_attention-mechanisms/attention.md
class DotProductAttention(nn.Block):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # query: (batch_size, #queries, d)
    # key: (batch_size, #kv_pairs, d)
    # value: (batch_size, #kv_pairs, dim_v)
    # valid_len: either (batch_size, ) or (batch_size, xx)
    def forward(self, query, key, value, valid_len=None):
        d = query.shape[-1]
        # Set transpose_b=True to swap the last two dimensions of key
        scores = npx.batch_dot(query, key, transpose_b=True) / math.sqrt(d)
        attention_weights = self.dropout(masked_softmax(scores, valid_len))
        return npx.batch_dot(attention_weights, value)


# Defined in file: ./chapter_attention-mechanisms/attention.md
class MLPAttention(nn.Block):
    def __init__(self, units, dropout, **kwargs):
        super(MLPAttention, self).__init__(**kwargs)
        # Use flatten=True to keep query's and key's 3-D shapes
        self.W_k = nn.Dense(units, activation='tanh',
                            use_bias=False, flatten=False)
        self.W_q = nn.Dense(units, activation='tanh',
                            use_bias=False, flatten=False)
        self.v = nn.Dense(1, use_bias=False, flatten=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, valid_len):
        query, key = self.W_k(query), self.W_q(key)
        # Expand query to (batch_size, #querys, 1, units), and key to
        # (batch_size, 1, #kv_pairs, units). Then plus them with broadcast
        features = np.expand_dims(query, axis=2) + np.expand_dims(key, axis=1)
        scores = np.squeeze(self.v(features), axis=-1)
        attention_weights = self.dropout(masked_softmax(scores, valid_len))
        return npx.batch_dot(attention_weights, value)


# Defined in file: ./chapter_attention-mechanisms/transformer.md
class MultiHeadAttention(nn.Block):
    def __init__(self, num_hiddens, num_heads, dropout, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Dense(num_hiddens, use_bias=False, flatten=False)
        self.W_k = nn.Dense(num_hiddens, use_bias=False, flatten=False)
        self.W_v = nn.Dense(num_hiddens, use_bias=False, flatten=False)
        self.W_o = nn.Dense(num_hiddens, use_bias=False, flatten=False)

    def forward(self, query, key, value, valid_len):
        # For self-attention, query, key, and value shape:
        # (batch_size, seq_len, dim), where seq_len is the length of input
        # sequence. valid_len shape is either (batch_size, ) or
        # (batch_size, seq_len).

        # Project and transpose query, key, and value from
        # (batch_size, seq_len, num_hiddens) to
        # (batch_size * num_heads, seq_len, num_hiddens / num_heads).
        query = transpose_qkv(self.W_q(query), self.num_heads)
        key = transpose_qkv(self.W_k(key), self.num_heads)
        value = transpose_qkv(self.W_v(value), self.num_heads)

        if valid_len is not None:
            # Copy valid_len by num_heads times
            if valid_len.ndim == 1:
                valid_len = np.tile(valid_len, self.num_heads)
            else:
                valid_len = np.tile(valid_len, (self.num_heads, 1))

        # For self-attention, output shape:
        # (batch_size * num_heads, seq_len, num_hiddens / num_heads)
        output = self.attention(query, key, value, valid_len)

        # output_concat shape: (batch_size, seq_len, num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


# Defined in file: ./chapter_attention-mechanisms/transformer.md
def transpose_qkv(X, num_heads):
    # Input X shape: (batch_size, seq_len, num_hiddens).
    # Output X shape:
    # (batch_size, seq_len, num_heads, num_hiddens / num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # X shape: (batch_size, num_heads, seq_len, num_hiddens / num_heads)
    X = X.transpose(0, 2, 1, 3)

    # output shape: (batch_size * num_heads, seq_len, num_hiddens / num_heads)
    output = X.reshape(-1, X.shape[2], X.shape[3])
    return output



# Defined in file: ./chapter_attention-mechanisms/transformer.md
def transpose_output(X, num_heads):
    # A reversed version of transpose_qkv
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.transpose(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


# Defined in file: ./chapter_attention-mechanisms/transformer.md
class PositionWiseFFN(nn.Block):
    def __init__(self, pw_num_hiddens, pw_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Dense(pw_num_hiddens, flatten=False,
                               activation='relu')
        self.dense2 = nn.Dense(pw_num_outputs, flatten=False)

    def forward(self, X):
        return self.dense2(self.dense1(X))


# Defined in file: ./chapter_attention-mechanisms/transformer.md
class AddNorm(nn.Block):
    def __init__(self, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm()

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


# Defined in file: ./chapter_attention-mechanisms/transformer.md
class PositionalEncoding(nn.Block):
    def __init__(self, embed_size, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = np.zeros((1, max_len, embed_size))
        X = np.arange(0, max_len).reshape(-1, 1) / np.power(
            10000, np.arange(0, embed_size, 2) / embed_size)
        self.P[:, :, 0::2] = np.sin(X)
        self.P[:, :, 1::2] = np.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].as_in_context(X.context)
        return self.dropout(X)


# Defined in file: ./chapter_attention-mechanisms/transformer.md
class EncoderBlock(nn.Block):
    def __init__(self, embed_size, pw_num_hiddens, num_heads, dropout,
                 **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(embed_size, num_heads, dropout)
        self.addnorm1 = AddNorm(dropout)
        self.ffn = PositionWiseFFN(pw_num_hiddens, embed_size)
        self.addnorm2 = AddNorm(dropout)

    def forward(self, X, valid_len):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_len))
        return self.addnorm2(Y, self.ffn(Y))


# Defined in file: ./chapter_attention-mechanisms/transformer.md
class TransformerEncoder(d2l.Encoder):
    def __init__(self, vocab_size, embed_size, pw_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoding = PositionalEncoding(embed_size, dropout)
        self.blks = nn.Sequential()
        for _ in range(num_layers):
            self.blks.add(
                EncoderBlock(embed_size, pw_num_hiddens, num_heads, dropout))

    def forward(self, X, valid_len, *args):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.embed_size))
        for blk in self.blks:
            X = blk(X, valid_len)
        return X


# Defined in file: ./chapter_optimization/optimization-intro.md
def annotate(text, xy, xytext):
    d2l.plt.gca().annotate(text, xy=xy, xytext=xytext,
                           arrowprops=dict(arrowstyle='->'))


# Defined in file: ./chapter_optimization/gd.md
def train_2d(trainer, steps=20):
    """Optimize a 2-dim objective function with a customized trainer."""
    # s1 and s2 are internal state variables and will
    # be used later in the chapter
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print('epoch %d, x1 %f, x2 %f' % (i + 1, x1, x2))
    return results


# Defined in file: ./chapter_optimization/gd.md
def show_trace_2d(f, results):
    """Show the trace of 2D variables during optimization."""
    d2l.set_figsize((3.5, 2.5))
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1))
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')


# Defined in file: ./chapter_optimization/minibatch-sgd.md
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')


# Defined in file: ./chapter_optimization/minibatch-sgd.md
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_iter = d2l.load_array(
        (data[:n, :-1], data[:n, -1]), batch_size, is_train=True)
    return data_iter, data.shape[1]-1


# Defined in file: ./chapter_optimization/minibatch-sgd.md
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Initialization
    w = np.random.normal(scale=0.01, size=(feature_dim, 1))
    b = np.zeros(1)
    w.attach_grad()
    b.attach_grad()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # Train
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print('loss: %.3f, %.3f sec/epoch' % (animator.Y[0][-1], timer.avg()))
    return timer.cumsum(), animator.Y[0]


# Defined in file: ./chapter_optimization/minibatch-sgd.md
def train_gluon_ch11(tr_name, hyperparams, data_iter, num_epochs=2):
    # Initialization
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))
    trainer = gluon.Trainer(net.collect_params(), tr_name, hyperparams)
    loss = gluon.loss.L2Loss()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(X.shape[0])
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print('loss: %.3f, %.3f sec/epoch' % (animator.Y[0][-1], timer.avg()))


# Defined in file: ./chapter_computational-performance/hybridize.md
class benchmark:    
    def __init__(self, description = 'Done in %.4f sec'):
        self.description = description
        
    def __enter__(self):
        self.timer = d2l.Timer()
        return self

    def __exit__(self, *args):
        print(self.description % self.timer.stop())


# Defined in file: ./chapter_computational-performance/multiple-gpus.md
def split_batch(X, y, ctx_list):
    """Split X and y into multiple devices specified by ctx."""
    assert X.shape[0] == y.shape[0]
    return (gluon.utils.split_and_load(X, ctx_list),
            gluon.utils.split_and_load(y, ctx_list))


# Defined in file: ./chapter_computational-performance/multiple-gpus-gluon.md
def resnet18(num_classes):
    """A slightly modified ResNet-18 model."""
    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(d2l.Residual(
                    num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(d2l.Residual(num_channels))
        return blk

    net = nn.Sequential()
    # This model uses a smaller convolution kernel, stride, and padding and
    # removes the maximum pooling layer
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'))
    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net


# Defined in file: ./chapter_computational-performance/multiple-gpus-gluon.md
def evaluate_accuracy_gpus(net, data_iter, split_f=d2l.split_batch):
    # Query the list of devices
    ctx = list(net.collect_params().values())[0].list_ctx()
    metric = d2l.Accumulator(2)  # num_corrected_examples, num_examples
    for features, labels in data_iter:
        Xs, ys = split_f(features, labels, ctx)
        pys = [net(X) for X in Xs]  # Run in parallel
        metric.add(sum(float(d2l.accuracy(py, y)) for py, y in zip(pys, ys)),
                   labels.size)
    return metric[0]/metric[1]


# Defined in file: ./chapter_computer-vision/image-augmentation.md
def train_batch_ch13(net, features, labels, loss, trainer, ctx_list,
                     split_f=d2l.split_batch):
    Xs, ys = split_f(features, labels, ctx_list)
    with autograd.record():
        pys = [net(X) for X in Xs]
        ls = [loss(py, y) for py, y in zip(pys, ys)]
    for l in ls:
        l.backward()
    trainer.step(labels.shape[0])
    train_loss_sum = sum([float(l.sum()) for l in ls])
    train_acc_sum = sum(d2l.accuracy(py, y) for py, y in zip(pys, ys))
    return train_loss_sum, train_acc_sum


# Defined in file: ./chapter_computer-vision/image-augmentation.md
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               ctx_list=d2l.try_all_gpus(), split_f=d2l.split_batch):
    num_batches, timer = len(train_iter), d2l.Timer()
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        # Store training_loss, training_accuracy, num_examples, num_features
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, ctx_list, split_f)
            metric.add(l, acc, labels.shape[0], labels.size)
            timer.stop()
            if (i+1) % (num_batches // 5) == 0:
                animator.add(epoch+i/num_batches,
                             (metric[0]/metric[2], metric[1]/metric[3], None))
        test_acc = d2l.evaluate_accuracy_gpus(net, test_iter, split_f)
        animator.add(epoch+1, (None, None, test_acc))
    print('loss %.3f, train acc %.3f, test acc %.3f' % (
        metric[0]/metric[2], metric[1]/metric[3], test_acc))
    print('%.1f examples/sec on %s' % (
        metric[2]*num_epochs/timer.sum(), ctx_list))


# Defined in file: ./chapter_computer-vision/fine-tuning.md
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL+'hotdog.zip', 
                         'fba480ffa8aa7e0febbb511d181409f899b9baa5')


# Defined in file: ./chapter_computer-vision/bounding-box.md
def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format."""
    # Convert the bounding box (top-left x, top-left y, bottom-right x,
    # bottom-right y) format to matplotlib format: ((upper-left x,
    # upper-left y), width, height)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)


# Defined in file: ./chapter_computer-vision/anchor.md
def show_bboxes(axes, bboxes, labels=None, colors=None):
    """Show bounding boxes."""
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj
    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(bbox.asnumpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))


# Defined in file: ./chapter_computer-vision/object-detection-dataset.md
d2l.DATA_HUB['pikachu'] = (d2l.DATA_URL + 'pikachu.zip',
                           '68ab1bd42143c5966785eb0d7b2839df8d570190')


# Defined in file: ./chapter_computer-vision/object-detection-dataset.md
def load_data_pikachu(batch_size, edge_size=256):
    """Load the pikachu dataset."""
    data_dir = d2l.download_extract('pikachu')
    train_iter = image.ImageDetIter(
        path_imgrec=data_dir + 'train.rec',
        path_imgidx=data_dir + 'train.idx',
        batch_size=batch_size,
        data_shape=(3, edge_size, edge_size),  # The shape of the output image
        shuffle=True,  # Read the dataset in random order
        rand_crop=1,  # The probability of random cropping is 1
        min_object_covered=0.95, max_attempts=200)
    val_iter = image.ImageDetIter(
        path_imgrec=data_dir + 'val.rec', batch_size=batch_size,
        data_shape=(3, edge_size, edge_size), shuffle=False)
    return train_iter, val_iter


# Defined in file: ./chapter_computer-vision/semantic-segmentation-and-dataset.md
d2l.DATA_HUB['voc2012'] = (d2l.DATA_URL + 'VOCtrainval_11-May-2012.tar',
                           '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')


# Defined in file: ./chapter_computer-vision/semantic-segmentation-and-dataset.md
def read_voc_images(voc_dir, is_train=True):
    """Read all VOC feature and label images."""
    txt_fname = '%s/ImageSets/Segmentation/%s' % (
        voc_dir, 'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [None] * len(images), [None] * len(images)
    for i, fname in enumerate(images):
        features[i] = image.imread('%s/JPEGImages/%s.jpg' % (voc_dir, fname))
        labels[i] = image.imread(
            '%s/SegmentationClass/%s.png' % (voc_dir, fname))
    return features, labels


# Defined in file: ./chapter_computer-vision/semantic-segmentation-and-dataset.md
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]


# Defined in file: ./chapter_computer-vision/semantic-segmentation-and-dataset.md
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']


# Defined in file: ./chapter_computer-vision/semantic-segmentation-and-dataset.md
def build_colormap2label():
    """Build an RGB color to label mapping for segmentation."""
    colormap2label = np.zeros(256 ** 3)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[(colormap[0]*256 + colormap[1])*256 + colormap[2]] = i
    return colormap2label


# Defined in file: ./chapter_computer-vision/semantic-segmentation-and-dataset.md
def voc_label_indices(colormap, colormap2label):
    """Map an RGB color to a label."""
    colormap = colormap.astype(np.int32)
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]


# Defined in file: ./chapter_computer-vision/semantic-segmentation-and-dataset.md
def voc_rand_crop(feature, label, height, width):
    """Randomly crop for both feature and label images."""
    feature, rect = image.random_crop(feature, (width, height))
    label = image.fixed_crop(label, *rect)
    return feature, label


# Defined in file: ./chapter_computer-vision/semantic-segmentation-and-dataset.md
class VOCSegDataset(gluon.data.Dataset):
    """A customized dataset to load VOC dataset."""

    def __init__(self, is_train, crop_size, voc_dir):
        self.rgb_mean = np.array([0.485, 0.456, 0.406])
        self.rgb_std = np.array([0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = build_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return (img.astype('float32') / 255 - self.rgb_mean) / self.rgb_std

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[0] >= self.crop_size[0] and
            img.shape[1] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature.transpose(2, 0, 1),
                voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)


# Defined in file: ./chapter_computer-vision/semantic-segmentation-and-dataset.md
def load_data_voc(batch_size, crop_size):
    """Download and load the VOC2012 semantic dataset."""
    voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
    num_workers = d2l.get_dataloader_workers()
    train_iter = gluon.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, last_batch='discard', num_workers=num_workers)
    test_iter = gluon.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        last_batch='discard', num_workers=num_workers)
    return train_iter, test_iter


# Defined in file: ./chapter_computer-vision/kaggle-gluon-cifar10.md
d2l.DATA_HUB['cifar10_tiny'] = (d2l.DATA_URL + 'kaggle_cifar10_tiny.zip',
                                '2068874e4b9a9f0fb07ebe0ad2b29754449ccacd')


# Defined in file: ./chapter_computer-vision/kaggle-gluon-cifar10.md
def read_csv_labels(fname):
    """Read fname to return a name to label dictionary."""
    with open(fname, 'r') as f:
        # Skip the file header line (column name)
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))


# Defined in file: ./chapter_computer-vision/kaggle-gluon-cifar10.md
def copyfile(filename, target_dir):
    """Copy a file into a target directory."""
    d2l.mkdir_if_not_exist(target_dir)
    shutil.copy(filename, target_dir)


# Defined in file: ./chapter_computer-vision/kaggle-gluon-cifar10.md
def reorg_train_valid(data_dir, labels, valid_ratio):
    # The number of examples of the class with the least examples in the
    # training dataset
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # The number of examples per class for the validation set
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file in os.listdir(data_dir + 'train'):
        label = labels[train_file.split('.')[0]]
        fname = data_dir + 'train/' + train_file
        # Copy to train_valid_test/train_valid with a subfolder per class
        copyfile(fname, data_dir + 'train_valid_test/train_valid/' + label)
        if label not in label_count or label_count[label] < n_valid_per_label:
            # Copy to train_valid_test/valid
            copyfile(fname, data_dir + 'train_valid_test/valid/' + label)
            label_count[label] = label_count.get(label, 0) + 1
        else:
            # Copy to train_valid_test/train
            copyfile(fname, data_dir+'train_valid_test/train/' + label)
    return n_valid_per_label


# Defined in file: ./chapter_computer-vision/kaggle-gluon-cifar10.md
def reorg_test(data_dir):
    for test_file in os.listdir(data_dir + 'test'):
        copyfile(data_dir + 'test/' + test_file, 
                 data_dir + 'train_valid_test/test/unknown/')


# Defined in file: ./chapter_computer-vision/kaggle-gluon-dog.md
d2l.DATA_HUB['dog_tiny'] = (d2l.DATA_URL + 'kaggle_dog_tiny.zip',
                            '7c9b54e78c1cedaa04998f9868bc548c60101362')


# Defined in file: ./chapter_natural-language-processing-pretraining/word2vec-dataset.md
d2l.DATA_HUB['ptb'] = (d2l.DATA_URL + 'ptb.zip', 
                       '319d85e578af0cdc590547f26231e4e31cdf1e42')


# Defined in file: ./chapter_natural-language-processing-pretraining/word2vec-dataset.md
def read_ptb():
    data_dir = d2l.download_extract('ptb')
    with open(data_dir + 'ptb.train.txt') as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]


# Defined in file: ./chapter_natural-language-processing-pretraining/word2vec-dataset.md
def subsampling(sentences, vocab):
    # Map low frequency words into <unk>
    sentences = [[vocab.idx_to_token[vocab[tk]] for tk in line]
                 for line in sentences]
    # Count the frequency for each word
    counter = d2l.count_corpus(sentences)
    num_tokens = sum(counter.values())

    # Return True if to keep this token during subsampling
    def keep(token):
        return(random.uniform(0, 1) <
               math.sqrt(1e-4 / counter[token] * num_tokens))

    # Now do the subsampling
    return [[tk for tk in line if keep(tk)] for line in sentences]


# Defined in file: ./chapter_natural-language-processing-pretraining/word2vec-dataset.md
def get_centers_and_contexts(corpus, max_window_size):
    centers, contexts = [], []
    for line in corpus:
        # Each sentence needs at least 2 words to form a
        # "central target word - context word" pair
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):  # Context window centered at i
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size),
                                 min(len(line), i + 1 + window_size)))
            # Exclude the central target word from the context words
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts


# Defined in file: ./chapter_natural-language-processing-pretraining/word2vec-dataset.md
class RandomGenerator(object):
    """Draw a random int in [0, n] according to n sampling weights."""
    def __init__(self, sampling_weights):
        self.population = list(range(len(sampling_weights)))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            self.candidates = random.choices(
                self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i-1]


# Defined in file: ./chapter_natural-language-processing-pretraining/word2vec-dataset.md
def get_negatives(all_contexts, corpus, K):
    counter = d2l.count_corpus(corpus)
    sampling_weights = [counter[i]**0.75 for i in range(len(counter))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # Noise words cannot be context words
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives


# Defined in file: ./chapter_natural-language-processing-pretraining/word2vec-dataset.md
def batchify(data):
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (np.array(centers).reshape(-1, 1), np.array(contexts_negatives),
            np.array(masks), np.array(labels))


# Defined in file: ./chapter_natural-language-processing-pretraining/word2vec-dataset.md
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled = subsampling(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(all_contexts, corpus, num_noise_words)
    dataset = gluon.data.ArrayDataset(
        all_centers, all_contexts, all_negatives)
    data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True,
                                      batchify_fn=batchify)
    return data_iter, vocab


# Defined in file: ./chapter_natural-language-processing-pretraining/bert.md
class BERTEncoder(nn.Block):
    def __init__(self, vocab_size, embed_size, pw_num_hiddens, num_heads,
                 num_layers, dropout, **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.segment_embedding = nn.Embedding(2, embed_size)
        self.pos_encoding = d2l.PositionalEncoding(embed_size, dropout)
        self.blks = nn.Sequential()
        for _ in range(num_layers):
            self.blks.add(d2l.EncoderBlock(
                embed_size, pw_num_hiddens, num_heads, dropout))

    def forward(self, tokens, segments, valid_len):
        # Shape of X remains unchanged in the following code snippet:
        # (batch size, max sequence length, embed_size)
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = self.pos_encoding(X)
        for blk in self.blks:
            X = blk(X, valid_len)
        return X


# Defined in file: ./chapter_natural-language-processing-pretraining/bert.md
class MaskLM(nn.Block):
    def __init__(self, vocab_size, num_hiddens, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential()
        self.mlp.add(
            nn.Dense(num_hiddens, flatten=False, activation='relu'))
        self.mlp.add(nn.LayerNorm())
        self.mlp.add(nn.Dense(vocab_size, flatten=False))

    def forward(self, X, masked_positions):
        num_masked_positions = masked_positions.shape[1]
        masked_positions = masked_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = np.arange(0, batch_size)
        # Suppose that batch_size = 2, num_masked_positions = 3,
        # batch_idx = np.array([0, 0, 0, 1, 1, 1])
        batch_idx = np.repeat(batch_idx, num_masked_positions)
        masked_X = X[batch_idx, masked_positions]
        masked_X = masked_X.reshape((batch_size, num_masked_positions, -1))
        masked_preds = self.mlp(masked_X)
        return masked_preds


# Defined in file: ./chapter_natural-language-processing-pretraining/bert.md
class NextSentencePred(nn.Block):
    def __init__(self, num_hiddens, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.mlp = nn.Sequential()
        self.mlp.add(nn.Dense(num_hiddens, activation='tanh'))
        self.mlp.add(nn.Dense(2))

    def forward(self, X):
        # 0 is the index of the CLS token
        X = X[:, 0, :]
        # X shape: (batch size, embed_size)
        return self.mlp(X)


# Defined in file: ./chapter_natural-language-processing-pretraining/bert.md
class BERTModel(nn.Block):
    def __init__(self, vocab_size, embed_size, pw_num_hiddens, num_heads,
                 num_layers, dropout):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, embed_size, pw_num_hiddens,
                                   num_heads, num_layers, dropout)
        self.nsp = NextSentencePred(embed_size)
        self.mlm = MaskLM(vocab_size, embed_size)

    def forward(self, tokens, segments, valid_len=None,
                masked_positions=None):
        X = self.encoder(tokens, segments, valid_len)
        if masked_positions is not None:
            mlm_Y = self.mlm(X, masked_positions)
        else:
            mlm_Y = None
        nsp_Y = self.nsp(X)
        return X, mlm_Y, nsp_Y


# Defined in file: ./chapter_natural-language-processing-pretraining/bert-pretraining.md
d2l.DATA_HUB['wikitext-2'] = (
    'https://s3.amazonaws.com/research.metamind.io/wikitext/'
    'wikitext-2-v1.zip', '3c914d17d80b1459be871a5039ac23e752a53cbe')


# Defined in file: ./chapter_natural-language-processing-pretraining/bert-pretraining.md
def read_wiki(data_dir):
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r') as f:
        lines = f.readlines()
    # A line represents a paragragh.
    paragraghs = [line.strip().lower().split(' . ')
                  for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraghs)
    return paragraghs


# Defined in file: ./chapter_natural-language-processing-pretraining/bert-pretraining.md
def get_next_sentence(sentence, next_sentence, paragraphs):
    if random.random() < 0.5:
        is_next = True
    else:
        # paragraphs is a list of lists of lists
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next


# Defined in file: ./chapter_natural-language-processing-pretraining/bert-pretraining.md
def get_tokens_and_segments(tokens_a, tokens_b):
    tokens = ['<cls>'] + tokens_a + ['<sep>'] + tokens_b + ['<sep>']
    segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
    return tokens, segment_ids


# Defined in file: ./chapter_natural-language-processing-pretraining/bert-pretraining.md
def get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs)
        # Consider 1 '<cls>' token and 2 '<sep>' tokens
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
             continue
        tokens, segment_ids = get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segment_ids, is_next))
    return nsp_data_from_paragraph


# Defined in file: ./chapter_natural-language-processing-pretraining/bert-pretraining.md
def replace_mlm_tokens(tokens, candidate_mlm_pred_positions, num_mlm_preds,
                       vocab):
    # Make a new copy of tokens for the input of a masked language model,
    # where the input may contain replaced '<mask>' or random tokens
    mlm_input_tokens = [token for token in tokens]
    mlm_pred_positions_and_labels = []
    # Shuffle for gettting 15% random tokens for prediction in the masked
    # language model task
    random.shuffle(candidate_mlm_pred_positions)
    for mlm_pred_position in candidate_mlm_pred_positions:
        if len(mlm_pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # 80% of the time: replace the word with the '<mask>' token
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # 10% of the time: keep the word unchanged
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # 10% of the time: replace the word with a random word
            else:
                masked_token = random.randint(0, len(vocab) - 1)
        mlm_input_tokens[mlm_pred_position] = masked_token
        mlm_pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, mlm_pred_positions_and_labels


# Defined in file: ./chapter_natural-language-processing-pretraining/bert-pretraining.md
def get_mlm_data_from_tokens(tokens, vocab):
    candidate_mlm_pred_positions = []
    # tokens is a list of strings
    for i, token in enumerate(tokens):
        # Special tokens are not predicted in the masked language model task
        if token in ['<cls>', '<sep>']:
            continue
        candidate_mlm_pred_positions.append(i)
    # 15% of random tokens will be predicted in the masked language model task
    num_mlm_preds = max(1, int(round(len(tokens) * 0.15)))
    mlm_input_tokens, mlm_pred_positions_and_labels = replace_mlm_tokens(
        tokens, candidate_mlm_pred_positions, num_mlm_preds, vocab)
    mlm_pred_positions_and_labels = sorted(mlm_pred_positions_and_labels,
                                           key=lambda x: x[0])
    # e.g., [[1, 'an'], [12, 'car'], [25, '<unk>']] -> [1, 12, 25]
    mlm_pred_positions = list(list(zip(*mlm_pred_positions_and_labels))[0])
    # e.g., [[1, 'an'], [12, 'car'], [25, '<unk>']] -> ['an', 'car', '<unk>']
    mlm_pred_labels = list(list(zip(*mlm_pred_positions_and_labels))[1])
    return vocab[mlm_input_tokens], mlm_pred_positions, vocab[mlm_pred_labels]


# Defined in file: ./chapter_natural-language-processing-pretraining/bert-pretraining.md
def pad_bert_inputs(instances, max_len, vocab):
    max_num_mlm_preds = int(round(max_len * 0.15))
    X_mlm_tokens, Y_mlm, X_mlm_pred_positions = [], [], [] 
    X_mlm_weights, Y_nsp, X_segments, valid_lens = [], [], [], []
    for (mlm_input_ids, mlm_pred_positions, mlm_pred_label_ids, segment_ids,
         is_next) in instances:
        X_mlm_tokens.append(np.array(mlm_input_ids + [vocab['<pad>']] * (
            max_len - len(mlm_input_ids)), dtype='int32'))
        Y_mlm.append(np.array(mlm_pred_label_ids + [0] * (
            20 - len(mlm_pred_label_ids)), dtype='int32'))
        X_mlm_pred_positions.append(np.array(mlm_pred_positions + [0] * (
            20 - len(mlm_pred_positions)), dtype='int32'))
        # Predictions of padded tokens will be filtered out in the loss via
        # multiplication of 0 weights
        X_mlm_weights.append(np.array([1.0] * len(mlm_pred_label_ids) + [
            0.0] * (20 - len(mlm_pred_positions)), dtype='float32'))
        Y_nsp.append(np.array(is_next))
        X_segments.append(np.array(segment_ids + [0] * (
            max_len - len(segment_ids)), dtype='int32'))
        valid_lens.append(np.array(len(mlm_input_ids)))
    return (X_mlm_tokens, Y_mlm, X_mlm_pred_positions, X_mlm_weights, Y_nsp,
            X_segments, valid_lens)


# Defined in file: ./chapter_natural-language-processing-pretraining/bert-pretraining.md
class WikiTextDataset(gluon.data.Dataset):
    def __init__(self, paragraghs, max_len=128):
        # Input paragraghs[i] is a list of sentence strings representing a
        # paragraph; while output paragraghs[i] is a list of sentences
        # representing a paragraph, where each sentence is a list of tokens
        paragraghs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraghs]
        sentences = [sentence for paragraph in paragraghs
                     for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # Get data for the next sentence prediction task
        instances = []
        for paragraph in paragraghs:
            instances.extend(get_nsp_data_from_paragraph(
                paragraph, paragraghs, self.vocab, max_len))
        # Get data for the masked language model task 
        instances = [(get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segment_ids, is_next))
                     for tokens, segment_ids, is_next in instances]
        # Pad inputs
        (self.X_mlm_tokens, self.Y_mlm, self.X_mlm_pred_positions,
         self.X_mlm_weights, self.Y_nsp, self.X_segments,
         self.valid_lens) = pad_bert_inputs(instances, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.X_mlm_tokens[idx], self.Y_mlm[idx],
                self.X_mlm_pred_positions[idx], self.X_mlm_weights[idx],
                self.Y_nsp[idx], self.X_segments[idx], self.valid_lens[idx])

    def __len__(self):
        return len(self.X_mlm_tokens)


# Defined in file: ./chapter_natural-language-processing-pretraining/bert-pretraining.md
def load_data_wiki(batch_size, max_len):
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraghs = read_wiki(data_dir)
    train_set = WikiTextDataset(paragraghs, max_len)
    train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True)
    return train_iter, train_set.vocab


# Defined in file: ./chapter_natural-language-processing-pretraining/bert-pretraining.md
def _get_batch_bert(batch, ctx):
    (X_mlm_tokens, Y_mlm, X_mlm_pred_positions, X_mlm_weights, Y_nsp,
     X_segments, valid_lens) = batch
    split_and_load = gluon.utils.split_and_load
    return (split_and_load(X_mlm_tokens, ctx, even_split=False),
            split_and_load(Y_mlm, ctx, even_split=False),
            split_and_load(X_mlm_pred_positions, ctx, even_split=False),
            split_and_load(X_mlm_weights, ctx, even_split=False),
            split_and_load(Y_nsp, ctx, even_split=False),
            split_and_load(X_segments, ctx, even_split=False),
            split_and_load(valid_lens.astype('float32'), ctx,
                           even_split=False))


# Defined in file: ./chapter_natural-language-processing-pretraining/bert-pretraining.md
def batch_loss_bert(net, nsp_loss, mlm_loss, X_mlm_tokens, Y_mlm,
                    X_mlm_pred_positions, X_mlm_weights, Y_nsp,
                    X_segments, valid_lens, vocab_size):
    ls = []
    ls_mlm = []
    ls_nsp = []
    for i_id, m_id, m_pos, m_w, nsl, s_i, v_l in zip(
        X_mlm_tokens, Y_mlm, X_mlm_pred_positions, X_mlm_weights,
        Y_nsp, X_segments, valid_lens):
        num_masks = m_w.sum() + 1e-8
        _, decoded, classified = net(i_id, s_i, v_l.reshape(-1),m_pos)
        l_mlm = mlm_loss(decoded.reshape((-1, vocab_size)),m_id.reshape(-1),
                         m_w.reshape((-1, 1)))
        l_mlm = l_mlm.sum() / num_masks
        l_nsp = nsp_loss(classified, nsl)
        l_nsp = l_nsp.mean()
        l = l_mlm + l_nsp
        ls.append(l)
        ls_mlm.append(l_mlm)
        ls_nsp.append(l_nsp)
        npx.waitall()
    return ls, ls_mlm, ls_nsp


# Defined in file: ./chapter_natural-language-processing-pretraining/bert-pretraining.md
def train_bert(data_eval, net, nsp_loss, mlm_loss, vocab_size, ctx,
               log_interval, max_step):
    trainer = gluon.Trainer(net.collect_params(), 'adam')
    step_num = 0
    while step_num < max_step:
        eval_begin_time = time.time()
        begin_time = time.time()

        running_mlm_loss = running_nsp_loss = 0
        total_mlm_loss = total_nsp_loss = 0
        running_num_tks = 0
        for _, data_batch in enumerate(data_eval):
            (X_mlm_tokens, Y_mlm, X_mlm_pred_positions, X_mlm_weights, \
             Y_nsp, X_segments, valid_lens) = _get_batch_bert(
                data_batch, ctx)

            step_num += 1
            with autograd.record():
                ls, ls_mlm, ls_nsp = batch_loss_bert(
                    net, nsp_loss, mlm_loss, X_mlm_tokens, Y_mlm,
                    X_mlm_pred_positions, X_mlm_weights, Y_nsp,
                    X_segments, valid_lens, vocab_size)
            for l in ls:
                l.backward()

            trainer.step(1)

            running_mlm_loss += sum([l for l in ls_mlm])
            running_nsp_loss += sum([l for l in ls_nsp])

            if (step_num + 1) % (log_interval) == 0:
                total_mlm_loss += running_mlm_loss
                total_nsp_loss += running_nsp_loss
                begin_time = time.time()
                running_mlm_loss = running_nsp_loss = 0

        eval_end_time = time.time()
        if running_mlm_loss != 0:
            total_mlm_loss += running_mlm_loss
            total_nsp_loss += running_nsp_loss
        total_mlm_loss /= step_num
        total_nsp_loss /= step_num
        print('Eval mlm_loss={:.3f}\tnsp_loss={:.3f}\t'
                     .format(float(total_mlm_loss),
                             float(total_nsp_loss)))
        print('Eval cost={:.1f}s'.format(eval_end_time - eval_begin_time))


# Defined in file: ./chapter_natural-language-processing-applications/sentiment-analysis-and-dataset.md
d2l.DATA_HUB['aclImdb'] = (
    'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
    '01ada507287d82875905620988597833ad4e0903')


# Defined in file: ./chapter_natural-language-processing-applications/sentiment-analysis-and-dataset.md
def read_imdb(data_dir, is_train):
    data, labels = [], []
    for label in ['pos/', 'neg/']:
        folder_name = data_dir + ('train/' if is_train else 'test/') + label
        for file in os.listdir(folder_name):
            with open(folder_name + file, 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos/' else 0)
    return data, labels


# Defined in file: ./chapter_natural-language-processing-applications/sentiment-analysis-and-dataset.md
def load_data_imdb(batch_size, num_steps=500):
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = np.array([d2l.trim_pad(vocab[line], num_steps, vocab.unk)
                               for line in train_tokens])
    test_features = np.array([d2l.trim_pad(vocab[line], num_steps, vocab.unk)
                              for line in test_tokens])
    train_iter = d2l.load_array((train_features, train_data[1]), batch_size)
    test_iter = d2l.load_array((test_features, test_data[1]), batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab


# Defined in file: ./chapter_natural-language-processing-applications/sentiment-analysis-rnn.md
def predict_sentiment(net, vocab, sentence):
    sentence = np.array(vocab[sentence.split()], ctx=d2l.try_gpu())
    label = np.argmax(net(sentence.reshape(1, -1)), axis=1)
    return 'positive' if label == 1 else 'negative'


# Defined in file: ./chapter_natural-language-processing-applications/natural-language-inference-and-dataset.md
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')


# Defined in file: ./chapter_natural-language-processing-applications/natural-language-inference-and-dataset.md
def read_snli(data_dir, is_train):
    """Read the SNLI dataset into premises, hypotheses, and labels."""
    def extract_text(s):
        # Remove information that will not be used by us
        s = re.sub('\(', '', s) 
        s = re.sub('\)', '', s)
        # Substitute two or more consecutive whitespace with space
        s = re.sub('\s{2,}', ' ', s)
        return s.strip()
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    file_name = (data_dir + 'snli_1.0_'+ ('train' if is_train else 'test')
                 + '.txt')
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels


# Defined in file: ./chapter_natural-language-processing-applications/natural-language-inference-and-dataset.md
class SNLIDataset(gluon.data.Dataset):
    """A customized dataset to load the SNLI dataset."""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        p_tokens = d2l.tokenize(dataset[0])
        h_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(p_tokens + h_tokens, min_freq=5,
                                   reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self.pad(p_tokens)
        self.hypotheses = self.pad(h_tokens)
        self.labels = np.array(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def pad(self, lines):
        return np.array([d2l.trim_pad(self.vocab[line], self.num_steps, 
                                      self.vocab['<pad>']) for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)


# Defined in file: ./chapter_natural-language-processing-applications/natural-language-inference-and-dataset.md
def load_data_snli(batch_size, num_steps=50):
    """Download the SNLI dataset and return data iterators and vocabulary."""
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True)
    test_iter = gluon.data.DataLoader(test_set, batch_size, shuffle=False)
    return train_iter, test_iter, train_set.vocab


# Defined in file: ./chapter_natural-language-processing-applications/natural-language-inference-attention.md
def split_batch_multi_inputs(X, y, ctx_list):
    """Split multi-input X and y into multiple devices specified by ctx"""
    X = list(zip(*[gluon.utils.split_and_load(
        feature, ctx_list, even_split=False) for feature in X]))
    return (X, gluon.utils.split_and_load(y, ctx_list, even_split=False))


# Defined in file: ./chapter_natural-language-processing-applications/natural-language-inference-attention.md
def predict_snli(net, premise, hypothesis):
    premise = np.array(vocab[premise], ctx=d2l.try_gpu())
    hypothesis = np.array(vocab[hypothesis], ctx=d2l.try_gpu())
    label = np.argmax(net([premise.reshape((1, -1)),
                           hypothesis.reshape((1, -1))]), axis=1)
    return 'entailment' if label == 0 else 'contradiction' if label == 1 \
            else 'neutral'


