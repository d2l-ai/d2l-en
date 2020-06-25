# This file is generated automatically through:
#    d2lbook build lib
# Don't edit it directly

# Defined in file: ./chapter_preface/index.md
import collections
from collections import defaultdict
from matplotlib import pyplot as plt
from IPython import display
import math
import jax
import jax.numpy as np
import numpy as onp
import os
import pandas as pd
import random
import re
import shutil
import sys
import tarfile
import time
import zipfile
import requests
d2l = sys.modules[__name__]


# Defined in file: ./chapter_preliminaries/pandas.md
def mkdir_if_not_exist(path):  #@save
    if not isinstance(path, str):
        path = os.path.join(*path)
    if not os.path.exists(path):
        os.makedirs(path)


# Defined in file: ./chapter_preliminaries/calculus.md
def use_svg_display():  #@save
    """Use the svg format to display a plot in Jupyter."""
    display.set_matplotlib_formats('svg')


# Defined in file: ./chapter_preliminaries/calculus.md
def set_figsize(figsize=(3.5, 2.5)):  #@save
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
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot data instances."""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()

    # Return True if `X` (ndarray or list) has 1 axis
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
class Timer:  #@save
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated times."""
        return np.array(self.times).cumsum().tolist()


# Defined in file: ./chapter_linear-networks/linear-regression-scratch.md
def synthetic_data(w, b, num_examples, rng_key):  #@save
    """Generate y = X w + b + noise."""
    X = jax.random.normal(rng_key, (num_examples, len(w)))
    y = np.dot(X, w) + b
    y += jax.random.normal(rng_key, (y.shape)) * 0.01 # Jax.random only has a standard normal sampler so we need to scale,
                                                  # See explanation below
    return X, y


# Defined in file: ./chapter_linear-networks/linear-regression-scratch.md
def linreg(X, w, b):  #@save
    return np.dot(X, w) + b


# Defined in file: ./chapter_linear-networks/linear-regression-scratch.md
squared_loss = (lambda y_hat, y: np.mean((y_hat - y.reshape(y_hat.shape))**2)) #@save


# Defined in file: ./chapter_linear-networks/linear-regression-scratch.md
def sgd(params, grads, lr, batch_size):  #@save
    new_params = []
    for param, delta in zip(params, grads):
        new_params.append(param - (lr * delta/batch_size)) # Jax arrays are immutable so we need to return
                                                           # our changed parameters 
    return new_params[0], new_params[1]


# Defined in file: ./chapter_linear-networks/image-classification-dataset.md
def get_fashion_mnist_labels(labels):  #@save
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


# Defined in file: ./chapter_linear-networks/image-classification-dataset.md
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):        
        if 'asnumpy' in dir(img): img = img.asnumpy() 
        if 'numpy' in dir(img): img = img.numpy()
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


# Defined in file: ./chapter_linear-networks/image-classification-dataset.md
def get_dataloader_workers(num_workers=4):  #@save
    # 0 means no additional process is used to speed up the reading of data.
    if sys.platform.startswith('win'):
        return 0
    else:
        return num_workers


# Defined in file: ./chapter_linear-networks/image-classification-dataset.md
def data_loader(data, targets, b_size, rng_key): #@save
    if len(targets.shape) == 1:
        targets = targets[..., None]
    while True:
        # Generate new subkey every time to not sample the same minibatch data over and over
        rng_key, subkey = jax.random.split(rng_key)
        data = jax.random.permutation(subkey, data)
        targets = jax.random.permutation(subkey, targets) # Shuffle data and targets in unison by using the same key

        n_samples = data.shape[0]
        idxs = onp.random.choice(n_samples, size=batch_size, replace=False)
        yield data[idxs], targets[idxs]
    


# Defined in file: ./chapter_linear-networks/image-classification-dataset.md
def load_data_fashion_mnist(batch_size, resize=None): #@save
    """Download the Fashion-MNIST dataset and then load into memory."""
    # TODO: Resize
    rng_key = jax.random.PRNGKey(24)
    fashion_mnist = fetch_openml(name="Fashion-MNIST")
    mnist_train_x, mnist_test_x, mnist_train_y, mnist_test_y = train_test_split(fashion_mnist['data'],
                                                                            fashion_mnist['target'],
                                                                            test_size=10000,
                                                                         )
    return (
        data_loader(mnist_train_x, mnist_train_y, batch_size, rng_key), data_loader(
        mnist_test_x, mnist_test_y, batch_size, rng_key))


