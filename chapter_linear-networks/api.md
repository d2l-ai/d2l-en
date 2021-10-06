```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# The D2L APIs
:label:`sec_d2l_apis`

Linear regression is one of the simplest machine learning models. Training it,
however, uses many of the same components as other models in this book require.
Therefore, before diving into the details it is worth reviewing some of the
functionality of the D2L library used throughout this book. This will greatly
streamline the presentation and you might even want to use it in your projects.

At its core we have three classes: `Module` contains models, losses and optimization methods; `DataModule` provides data loaders for training and validation. Both classes are combined using the `Trainer` class. It allows us to
train models on a variety of hardware platforms. Most code in this book adapts `Module` and `DataModule`. We will touch upon the `Trainer` class only when we discuss GPUs, CPUs, parallel training and optimization algorithms.

```{.python .input}
%%tab mxnet
import time
import numpy as np
from d2l import mxnet as d2l
from mxnet.gluon import nn
```

```{.python .input}
%%tab pytorch
import time
import numpy as np
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
import time
import numpy as np
from d2l import torch as d2l
import tensorflow as tf
```

## Utilities

We need a few utilities to simplify object oriented programming in notebooks. One of the challenges is that class definitions tend to be fairly long blocks of code. Notebook readability demands short code fragments, interspersed with explanations, a requirement incompatible with the style of programming common for Python libraries. The first
utility function allows us to register functions as methods in a class *after* the class has been created. In fact, we can do so *even after* we've created instances of the class! It allows us to split the implementation of a class into multiple code blocks.

```{.python .input}
%%tab all
def add_to_class(Class):  #@save
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper
```

Let's have a quick look at how to use it. We plan to implement a class `A` with a method `do`. Instead of having code for both `A` and `do` in the same code block, we can first declare the class `A` and create an instance `a`.

```{.python .input}
%%tab all
class A:
    def __init__(self):
        self.b = 1

a = A()
```

Next we define the method `do` as we normally would, but not in class `A`'s scope. Instead, we decorate this function by `add_to_class` with class `A` as its argument. In doing so, the method is able to access the member variables of `A` as we would expect if it had been defined as part of `A`'s definition. Let's see what happens when we invoke it for the instance `a`.

```{.python .input}
%%tab all
@add_to_class(A)
def do(self):
    print('class attribute "b" is', self.b)

a.do()
```

The second one is a utility class that saves all arguments in a class's `__init__` method as class attributes. This allows us to extend constructor call signatures implicitly without additional code.

```{.python .input}
%%tab all
class HyperParameters:  #@save
    def save_hyperparameters(self, ignore=[]):
        raise NotImplemented
```

We defer its implementation into :numref:`sec_utils`. To use it, we subclass our class from `HyperParameters` and call `save_hyperparameters` in the `__init__` method.

```{.python .input}
%%tab all
class B(d2l.HyperParameters):  # call the one saved in d2l with code implementation.
    def __init__(self, a, b, c):
        self.save_hyperparameters(ignore=['c'])
        print('self.a =', self.a, 'self.b =', self.b)
        print('there is no self.c =', not hasattr(self, 'c'))

B(a=1, b=2, c=3);
```

The last utility allows us to plot experiment progress interactively while it is going on. In deference to the much more powerful (and complex) [Tensorboard](https://www.tensorflow.org/tensorboard) we name it `ProgressBoard`. The  implementation is deferred to :numref:`sec_utils`. For now, let's simply see it in action.

The `draw` function plots a point `(x, y)` in the figure, with `label` specific the legend. The optional `every_n` smooths the line by only showing $1/n$ points in the figure. Their values are averaged from the $n$ neighbor points in the original figure.

```{.python .input}
%%tab all
class ProgressBoard(d2l.HyperParameters):  #@save
    """Plot data points in animation."""
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        raise NotImplemented
```

The following example, we draw `sin` and `cos` with a different smoothness. If you run this code block, you will see the lines grow in an animation.

```{.python .input}
%%tab all
board = d2l.ProgressBoard('x')
for x in np.arange(0, 10, 0.1):
    board.draw(x, np.sin(x), 'sin', every_n=2)
    board.draw(x, np.cos(x), 'cos', every_n=10)
```

## Model

The `Module` class  is the base class of all models we will implement. At a minimum we need to define three methods. The `__init__` method stores the learnable parameters, the `training_step` method accepts a data batch to return the loss value, the `configure_optimizers` method returns the optimization method, or a list of them, that is used to update the learnable parameters. Optionally we can define `validation_step` to report the evaluation metrics.
Sometimes we put the code to compute the outputs into a separate `forward` method to make it more reusable.

```{.python .input}
%%tab all
class Module(d2l.nn_Module, d2l.HyperParameters):  #@save
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()
        if tab.selected('tensorflow'):
            self.training = None

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is defined'
        return self.net(X)

    if tab.selected('tensorflow'):
        def call(self, X, *args, training=None):
            if training is not None:
                self.training = training
            return self.forward(X, *args)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'trainer is not inited'
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
        if tab.selected('mxnet', 'tensorflow'):
            self.board.draw(x, d2l.numpy(value), (
                'train_' if train else 'val_') + key,every_n=int(n))
        if tab.selected('pytorch'):
            self.board.draw(x, d2l.numpy(d2l.to(value, d2l.cpu())),
                            ('train_' if train else 'val_') + key,
                            every_n=int(n))

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        raise NotImplementedError
```

:begin_tab:`mxnet`
You may notice that `Module` is a subclass of `nn.Block`, the base class of a neural network in Gluon.
It provides convenient features to handle neural networks. For example, if we define a `forward` method, such as `forward(self, X)`, then for an instance `a` we can invoke this function by `a(X)`. This works since it calls the `forward` method in the built-in `__call__` method. You can find more details and examples about `nn.Block` in :numref:`sec_model_construction`.
:end_tab:

:begin_tab:`pytorch`
You may notice that `Module` is a subclass of `nn.Module`, the base class of neural network in PyTorch.
It provides convenient features to handle neural networks. For example, if we define a `forward` method, such as `forward(self, X)`, then for an instance `a` we can invoke this function by `a(X)`. This works since it calls the `forward` method in the built-in `__call__` method. You can find more details and examples about `nn.Block` in :numref:`sec_model_construction`.
:end_tab:

:begin_tab:`tensorflow`
You may notice that `Module` is a subclass of `tf.keras.Model`, the base class of neural network in TensorFlow.
It provides convenient features to handle neural networks. For example, it invokes the `call` method in the built-in `__call__` method. Here we redirect `call` to the `forward` function, saving its arguments as a class attribute. We do this to make our code more similar to other framework implementations.
:end_tab:

##  Data

The `DataModule` class is the base class for data. Quite frequently the `__init__` method is used to prepare the data. This includes downloading and preprocessing if needed. The `train_dataloader` returns the data loader for the training dataset. A data loader is a (Python) generator that yields a data batch each time it is used. This batch is then fed into the `training_step` method of `Module` to compute the loss. There is an optional `val_dataloader` to return the validation dataset loader. It behaves in the same manner, except that it yields data batches for the `validation_step` method in `Module`.

```{.python .input}
%%tab all
class DataModule(d2l.HyperParameters):  #@save
    if tab.selected('mxnet', 'pytorch'):
        def __init__(self, root='../data', num_workers=4):
            self.save_hyperparameters()

    if tab.selected('tensorflow'):
        def __init__(self, root='../data'):
            self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)
```

## Training

The `Trainer` class trains the learnable parameters (aka weights) in the `Module` class with data specified in `DataModule`. The key method is `fit`, which accepts two arguments: `model`, an instance of `Module`, and `data`, an instance of `DataModule`. It then iterates over the data `max_epochs` times to train the model. As before, we will defer the implementation of this function to later chapters.

```{.python .input}
%%tab all
class Trainer(d2l.HyperParameters):  #@save
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

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def fit_epoch(self):
        raise NotImplementedError

```

## Summary

The classes provided by the D2L API function as a lightweight toolkit that make structured modeling for deep learning easy. In particular, it makes it easy to reuse many components between projects without changing much at all. For instance, we can replace just the optimizer, just the model, just the dataset, etc.; This degree of modularity pays dividends throughout the book in terms of conciseness and simplicity (this is why we added it) and it can do the same for your own projects. We strongly recommend that you look at the implementation in detail once you have gained some more familiarity with deep learning modeling.
