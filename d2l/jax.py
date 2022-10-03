DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

import jax
import flax
from jax import numpy as jnp
from flax import linen as nn
import random

get_seed = lambda: random.randint(0, 1e6)

nn_Module = nn.Module


#################   WARNING   ################
# The below part is generated automatically through:
#    d2lbook build lib
# Don't edit it directly

import collections
import hashlib
import inspect
import math
import os
import random
import re
import shutil
import sys
import tarfile
import time
import zipfile
from collections import defaultdict
import pandas as pd
import requests
from IPython import display
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline

d2l = sys.modules[__name__]

from dataclasses import field
import flax
import jax
import numpy as np
import optax
import torch  # Used for dataloading
import torchvision  # Used for dataloading
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import grad
from jax import numpy as jnp
from jax import vmap

def use_svg_display():
    """Use the svg format to display a plot in Jupyter.

    Defined in :numref:`sec_calculus`"""
    backend_inline.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size for matplotlib.

    Defined in :numref:`sec_calculus`"""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib.

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim),     axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

def plot(X, Y=None, xlabel=None, ylabel=None, legend=[], xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot data points.

    Defined in :numref:`sec_calculus`"""

    def has_one_axis(X):  # True if `X` (tensor or list) has 1 axis
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X): X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)

    set_figsize(figsize)
    if axes is None: axes = d2l.plt.gca()
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x,y,fmt) if len(x) else axes.plot(y,fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

def add_to_class(Class):
    """Defined in :numref:`sec_oo-design`"""
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper

class HyperParameters:
    def save_hyperparameters(self, ignore=[]):
        """Defined in :numref:`sec_oo-design`"""
        raise NotImplemented

    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes.
    
        Defined in :numref:`sec_utils`"""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k:v for k, v in local_vars.items()
                        if k not in set(ignore+['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)

class ProgressBoard(d2l.HyperParameters):
    """Plot data points in animation.

    Defined in :numref:`sec_oo-design`"""
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        raise NotImplemented

    def draw(self, x, y, label, every_n=1):
        """Defined in :numref:`sec_utils`"""
        Point = collections.namedtuple('Point', ['x', 'y'])
        if not hasattr(self, 'raw_points'):
            self.raw_points = collections.OrderedDict()
            self.data = collections.OrderedDict()
        if label not in self.raw_points:
            self.raw_points[label] = []
            self.data[label] = []
        points = self.raw_points[label]
        line = self.data[label]
        points.append(Point(x, y))
        if len(points) != every_n:
            return
        mean = lambda x: sum(x) / len(x)
        line.append(Point(mean([p.x for p in points]),
                          mean([p.y for p in points])))
        points.clear()
        if not self.display:
            return
        d2l.use_svg_display()
        if self.fig is None:
            self.fig = d2l.plt.figure(figsize=self.figsize)
        plt_lines, labels = [], []
        for (k, v), ls, color in zip(self.data.items(), self.ls, self.colors):
            plt_lines.append(d2l.plt.plot([p.x for p in v], [p.y for p in v],
                                          linestyle=ls, color=color)[0])
            labels.append(k)
        axes = self.axes if self.axes else d2l.plt.gca()
        if self.xlim: axes.set_xlim(self.xlim)
        if self.ylim: axes.set_ylim(self.ylim)
        if not self.xlabel: self.xlabel = self.x
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_xscale(self.xscale)
        axes.set_yscale(self.yscale)
        axes.legend(plt_lines, labels)
        display.display(self.fig)
        display.clear_output(wait=True)

class Module(d2l.nn_Module, d2l.HyperParameters):
    """Defined in :numref:`sec_oo-design`"""
    # No need for save_hyperparam when using Python dataclass
    plot_train_per_epoch: int = field(default=2, init=False)
    plot_valid_per_epoch: int = field(default=1, init=False)
    # Use default_factory to make sure new plots are generated on each run
    board: ProgressBoard = field(default_factory=lambda: ProgressBoard(),
                                 init=False)
    training: bool = field(default=None, init=False)

    def loss(self, y_hat, y):
        raise NotImplementedError

    # JAX & Flax don't have a forward-method-like syntax. Flax uses setup
    # and built-in __call__ magic methods for forward pass. Adding here
    # for consistency
    def forward(self, X, *args, **kwargs):
        assert hasattr(self, 'net'), 'Neural network is defined'
        return self.net(X, *args, **kwargs)

    def __call__(self, X, *args, **kwargs):
        return self.forward(X, *args, **kwargs)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        self.board.draw(x, d2l.to(value, d2l.cpu()),
                        ('train_' if train else 'val_') + key,
                        every_n=int(n))

    def training_step(self, params, batch):
        l, grads = jax.value_and_grad(self.loss)(params, *batch[:-1],
                                                 batch[-1])
        self.plot("loss", l, train=True)
        return l, grads

    def validation_step(self, params, batch):
        l = self.loss(params, *batch[:-1], batch[-1])
        self.plot('loss', l, train=False)

    def apply_init(self, dummy_input, **kwargs):
        """To be defined later in :numref:`sec_lazy_init`"""
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    def configure_optimizers(self):
        """Defined in :numref:`sec_classification`"""
        return optax.sgd(self.lr)

    def apply_init(self, dummy_input, **kwargs):
        """Defined in :numref:`sec_lazy_init`"""
        if kwargs and 'key' in kwargs and (kwargs['key'] is not None):
            self.key = kwargs['key']
        else:
            self.key = jax.random.PRNGKey(d2l.get_seed())
        params = self.init(self.key, dummy_input)
        return params

class DataModule(d2l.HyperParameters):
    """Defined in :numref:`sec_oo-design`"""
    def __init__(self, root='../data', num_workers=4):
        self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)

    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        """Defined in :numref:`sec_synthetic-regression-data`"""
        tensors = tuple(a[indices] for a in tensors)
        # Use PyTorch Dataset and Dataloader
        # JAX or Flax do not provide any dataloading functionality
    
        def jax_collate(batch):
            if isinstance(batch[0], np.ndarray):
                return jnp.stack(batch)
            elif isinstance(batch[0], (tuple, list)):
                transposed = zip(*batch)
                return [jax_collate(samples) for samples in transposed]
            else:
                return jnp.array(batch)
    
        class JaxDataset(torch.utils.data.Dataset):
            def __init__(self, train, seed=0):
                super().__init__()
    
            def __getitem__(self, index):
                return (np.asarray(tensors[0][index]),
                        np.asarray(tensors[1][index]))
    
            def __len__(self):
                return len(tensors[0])
    
        dataset = JaxDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size,
                                           shuffle=train,
                                           collate_fn=jax_collate)
    

class Trainer(d2l.HyperParameters):
    """Defined in :numref:`sec_oo-design`"""
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, 'No GPU support yet'

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    def fit(self, model, data, key=None):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        dummy_input = next(iter(self.train_dataloader))[0]
        # Flax uses optax under the hood for a single state obj TrainState
        self.state = TrainState.create(apply_fn=model.apply,
                                       params=model.apply_init(dummy_input, key=key),
                                       tx=model.configure_optimizers())
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def fit_epoch(self):
        raise NotImplementedError

    def prepare_batch(self, batch):
        """Defined in :numref:`sec_linear_scratch`"""
        return batch

    def fit_epoch(self):
        """Defined in :numref:`sec_linear_scratch`"""
        self.model.training = True
        for batch in self.train_dataloader:
            _, grads = self.model.training_step(self.state.params,
                                                self.prepare_batch(batch))
            self.state = self.state.apply_gradients(grads=grads)
            self.train_batch_idx += 1
        if self.val_dataloader is None:
            return
        self.model.training = False
        for batch in self.val_dataloader:
            self.model.validation_step(self.state.params,
                                       self.prepare_batch(batch))
            self.val_batch_idx += 1

    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        """Defined in :numref:`sec_use_gpu`"""
        self.save_hyperparameters()
        self.gpus = [d2l.gpu(i) for i in range(min(num_gpus, d2l.num_gpus()))]
    

    def prepare_batch(self, batch):
        """Defined in :numref:`sec_use_gpu`"""
        if self.gpus:
            batch = [d2l.to(a, self.gpus[0]) for a in batch]
        return batch

class SyntheticRegressionData(d2l.DataModule):
    """Defined in :numref:`sec_synthetic-regression-data`"""
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000,
                 batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        key = jax.random.PRNGKey(0)
        key1, key2 = jax.random.split(key)
        self.X = jax.random.normal(key1, (n, w.shape[0]))
        noise = jax.random.normal(key2, (n, 1)) * noise
        self.y = d2l.matmul(self.X, d2l.reshape(w, (-1, 1))) + b + noise

    def get_dataloader(self, train):
        """Defined in :numref:`sec_synthetic-regression-data`"""
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.X, self.y), train, i)

class LinearRegressionScratch(d2l.Module):
    """Defined in :numref:`sec_linear_scratch`"""
    num_inputs: int
    lr: float
    sigma: float = 0.01

    def setup(self):
        self.w = self.param('w', nn.initializers.normal(self.sigma),
                            (self.num_inputs, 1))
        self.b = self.param('b', nn.initializers.zeros, (1))

    def forward(self, X):
        """The linear regression model.
    
        Defined in :numref:`sec_linear_scratch`"""
        return d2l.matmul(X, self.w) + self.b

    def loss(self, params, X, y):
        """Defined in :numref:`sec_linear_scratch`"""
        y_hat = self.apply(params, X)
        l = (y_hat - d2l.reshape(y, y_hat.shape)) ** 2 / 2
        return d2l.reduce_mean(l)

    def configure_optimizers(self):
        """Defined in :numref:`sec_linear_scratch`"""
        return SGD(self.lr)

class SGD(d2l.HyperParameters):
    """Defined in :numref:`sec_linear_scratch`"""
    def __init__(self, lr):
        """
        Minibatch stochastic gradient descent.
        The key transformation of Optax is the GradientTransformation
        defined by two methods, the init and the update.
        The init initializes the state and the update transforms
        the gradients.
        https://github.com/deepmind/optax/blob/master/optax/_src/transform.py
        """
        self.save_hyperparameters()

    def init(self, params):
        # Delete unused params
        del params
        return optax.EmptyState

    def update(self, updates, state, params=None):
        del params
        # When state.apply_gradients method is called to update flax's
        # train_state object, it internally calls optax.apply_updates method
        # adding the params to the update equation defined below.
        updates = jax.tree_util.tree_map(lambda g: -self.lr * g, updates)
        return updates, state

    def __call__():
        return optax.GradientTransformation(self.init, self.update)

class LinearRegression(d2l.Module):
    """Defined in :numref:`sec_linear_concise`"""
    lr: float

    def setup(self):
        self.net = nn.Dense(1, kernel_init=nn.initializers.normal(0.01))

    def forward(self, X):
        """The linear regression model.
    
        Defined in :numref:`sec_linear_concise`"""
        return self.net(X)

    def loss(self, params, X, y):
        """Defined in :numref:`sec_linear_concise`"""
        y_hat = self.apply(params, X)
        return d2l.reduce_mean(optax.l2_loss(y_hat, y))

    def configure_optimizers(self):
        """Defined in :numref:`sec_linear_concise`"""
        return optax.sgd(self.lr)

    def get_w_b(self, state):
        """Defined in :numref:`sec_linear_concise`"""
        net = state.params['params']['net']
        return net['kernel'], net['bias']

class ToArray:
    """Convert a PIL Image to numpy.ndarray."""
    def __init__(self):
        """Defined in :numref:`sec_fashion_mnist`"""
        pass

    def __call__(self, img):
        return np.asarray(img) / 255  # Normalize arrays

class FashionMNIST(d2l.DataModule):
    """Defined in :numref:`sec_fashion_mnist`"""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = torchvision.transforms.Compose(
                                    [torchvision.transforms.Resize(resize),
                                     ToArray()])
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=trans, download=True)
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=trans, download=True)

    def text_labels(self, indices):
        """Return text labels.
    
        Defined in :numref:`sec_fashion_mnist`"""
        labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                  'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [labels[int(i)] for i in indices]

    def get_dataloader(self, train):
        """Defined in :numref:`sec_fashion_mnist`"""
        def jax_collate(batch):
            if isinstance(batch[0], np.ndarray):
                return jnp.stack(batch)
            elif isinstance(batch[0], (tuple, list)):
                transposed = zip(*batch)
                return [jax_collate(samples) for samples in transposed]
            else:
                return jnp.array(batch)
        data = self.train if train else self.val
        return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train,
                                           collate_fn=jax_collate, num_workers=0)

    def visualize(self, batch, nrows=1, ncols=8, labels=[]):
        """Defined in :numref:`sec_fashion_mnist`"""
        X, y = batch
        if not labels:
            labels = self.text_labels(y)
        d2l.show_images(jnp.squeeze(X), nrows, ncols, titles=labels)

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images.

    Defined in :numref:`sec_fashion_mnist`"""
    raise NotImplementedError

class Classifier(d2l.Module):
    """Defined in :numref:`sec_classification`"""
    def validation_step(self, params, batch):
        self.plot('loss', self.loss(params, *batch[:-1], batch[-1]),
                  train=False)
        self.plot('acc', self.accuracy(params, *batch[:-1], batch[-1]),
                  train=False)

    def accuracy(self, params, X, Y, averaged=True):
        """Compute the number of correct predictions.
    
        Defined in :numref:`sec_classification`"""
        Y_hat = self.apply(params, X)
        Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
        preds = d2l.astype(d2l.argmax(Y_hat, axis=1), Y.dtype)
        compare = d2l.astype(preds == d2l.reshape(Y, -1), d2l.float32)
        return d2l.reduce_mean(compare) if averaged else compare

    def loss(self, params, X, Y, averaged=True):
        """Defined in :numref:`sec_softmax_concise`"""
        Y_hat = self.apply(params, X)
        Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
        fn = optax.softmax_cross_entropy_with_integer_labels
        return fn(Y_hat, Y).mean() if averaged else fn(Y_hat, Y)

    def layer_summary(self, X_shape, key=jax.random.PRNGKey(d2l.get_seed())):
        """Defined in :numref:`sec_lenet`"""
        X = jnp.zeros(X_shape)
        tabulate_fn = nn.tabulate(self, key, method=self.forward)
        print(tabulate_fn(X))

def cpu():
    """Defined in :numref:`sec_use_gpu`"""
    return jax.devices('cpu')[0]

def gpu(i=0):
    """Defined in :numref:`sec_use_gpu`"""
    return jax.devices('gpu')[i]

def num_gpus():
    """Defined in :numref:`sec_use_gpu`"""
    return jax.device_count() - 1  # Exclude CPU device

def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().

    Defined in :numref:`sec_use_gpu`"""
    if num_gpus() >= i + 1:
        return gpu(i)
    return cpu()

def try_all_gpus():
    """Return all available GPUs, or [cpu(),] if no GPU exists.

    Defined in :numref:`sec_use_gpu`"""
    return [gpu(i) for i in range(num_gpus())]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images.

    Defined in :numref:`sec_utils`"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        try:
            img = d2l.numpy(img)
        except:
            pass
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

def download(url, folder='../data', sha1_hash=None):
    """Download a file to folder and return the local filepath.

    Defined in :numref:`sec_utils`"""
    if not url.startswith('http'):
        # For back compatability
        url, sha1_hash = DATA_HUB[url]
    os.makedirs(folder, exist_ok=True)
    fname = os.path.join(folder, url.split('/')[-1])
    # Check if hit cache
    if os.path.exists(fname) and sha1_hash:
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    # Download
    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def extract(filename, folder=None):
    """Extract a zip/tar file into folder.

    Defined in :numref:`sec_utils`"""
    base_dir = os.path.dirname(filename)
    _, ext = os.path.splitext(filename)
    assert ext in ('.zip', '.tar', '.gz'), 'Only support zip/tar files.'
    if ext == '.zip':
        fp = zipfile.ZipFile(filename, 'r')
    else:
        fp = tarfile.open(filename, 'r')
    if folder is None:
        folder = base_dir
    fp.extractall(folder)


# Alias defined in config.ini
nn_Module = nn.Module
to = jax.device_put
numpy = np.asarray

ones_like = jnp.ones_like
ones = jnp.ones
zeros_like = jnp.zeros_like
zeros = jnp.zeros
arange = jnp.arange
meshgrid = jnp.meshgrid
sin = jnp.sin
sinh = jnp.sinh
cos = jnp.cos
cosh = jnp.cosh
tanh = jnp.tanh
linspace = jnp.linspace
exp = jnp.exp
log = jnp.log
tensor = jnp.array
expand_dims = jnp.expand_dims
matmul = jnp.matmul
int32 = jnp.int32
int64 = jnp.int64
float32 = jnp.float32
concat = jnp.concatenate
stack = jnp.stack
abs = jnp.abs
eye = jnp.eye
reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.astype(*args, **kwargs)
reduce_mean = lambda x, *args, **kwargs: x.mean(*args, **kwargs)
swapaxes = lambda x, *args, **kwargs: x.swapaxes(*args, **kwargs)
repeat = lambda x, *args, **kwargs: x.repeat(*args, **kwargs)

