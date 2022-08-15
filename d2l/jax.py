DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

import jax
import flax
import jax.numpy as jnp
from flax import linen as nn

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
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import grad, random, vmap

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
    # No need for save_hyperparam when using python dataclass
    # python dataclasses do not work perfectly well with inheritance
    plot_train_per_epoch: int = field(default=2, init=False)
    plot_valid_per_epoch: int = field(default=1, init=False)
    # Use default_factory to make sure new plots are generated on each run
    board: ProgressBoard = field(default_factory=lambda: ProgressBoard(),
                                 init=False)
    training: bool = field(default=None, init=False)

    def loss(self, y_hat, y):
        raise NotImplementedError

    # JAX & Flax don't have a forward method like syntax
    # Flax uses setup and built-in __call__ magic methods for forward pass
    # Adding here for consistency
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
        self.plot("loss", l, train=False)

    def configure_optimizers(self):
        raise NotImplementedError

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
        from torch.utils import data
    
        def jax_collate(batch):
            if isinstance(batch[0], np.ndarray):
                return jnp.stack(batch)
            elif isinstance(batch[0], (tuple,list)):
                transposed = zip(*batch)
                return [jax_collate(samples) for samples in transposed]
            else:
                return jnp.array(batch)
    
        class JaxDataset(data.Dataset):
            def __init__(self, train, seed=0):
                super().__init__()
    
            def __getitem__(self, index):
                return (np.asarray(tensors[0][index]),
                        np.asarray(tensors[1][index]))
    
            def __len__(self):
                return len(tensors[0])
    
        dataset = JaxDataset(*tensors)
        return data.DataLoader(dataset, self.batch_size,
                               shuffle=train, collate_fn=jax_collate)
    

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

    def apply_init(self, **kwargs):
        if kwargs and 'key' in kwargs and (kwargs['key'] is not None):
            self.key = kwargs['key']
        else:
            self.key = jax.random.PRNGKey(0)  # Avoid, but use as fallback
        input_shape = next(iter(self.train_dataloader))[0].shape
        dummy_input = jnp.zeros(input_shape)
        params = self.model.init(self.key, dummy_input)
        return params

    def fit(self, model, data, key=None):
        self.prepare_data(data)
        self.prepare_model(model)
        self.params = self.apply_init(key=key)
        self.optim = model.configure_optimizers()
        # Flax uses optax under the hood for a single state obj TrainState
        self.state = TrainState.create(apply_fn=model.apply,
                                       params=self.apply_init(key=key),
                                       tx=model.configure_optimizers())
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def fit_epoch(self):
        raise NotImplementedError

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

def cpu():
    """Defined in :numref:`sec_use_gpu`"""
    return jax.devices('cpu')[0]

def gpu(i=0):
    """Defined in :numref:`sec_use_gpu`"""
    return jax.devices('gpu')[i]

def num_gpus():
    """Defined in :numref:`sec_use_gpu`"""
    return jax.device_count('gpu')


# Alias defined in config.ini
nn_Module = nn.Module
to = jax.device_put

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

