```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# The D2L APIs
:label:`sec_d2l_apis`

The linear regression model is one of the simplest machine learning models.
Training this model, however, shares the same components as other model trainings in the rest of this book.
Therefore,
before diving into the code implementation,
let us first explain how we organize our code through this book.
It will make you easier to read our code and even use it in your projects.

The foundation of our code consists of three classes: `Module` for models, losses and optimization methods, `DataModule` for training and validation data loaders, and `Trainer` glues `Module` and `DataModule` to train the models on various hardware. Most code in the rest of this book is in subclasses of `Module` and `DataModule`.

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


Let's first introduce utility functions and classes.
We will adopt the object-oriented programming that is common for Python libraries. It's, however, used less in notebooks where we often keep a code block short for readability. The first utility function allows to register a function to a class defined in a previous code block. So now we can split the implementation of a class into multiple code blocks.

```{.python .input}
%%tab all
def add_to_class(Class):  #@save
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper
```

Let's give an example for how to use it. If we plan to implement a class `A` with a method `do`. Instead of having code for both `A` and `do` in the same code block, we can first declare the class `A` and construct an instance `a`.

```{.python .input}
%%tab all
class A:
    def __init__(self):
        self.a = 1

a = A()
```

Next we define the class method `do` as we do normally but not in the class `A`'s scope. Instead, we decorate this function by `add_to_class` with class `A` as its argument. Then we can see that the instance `a` we created in the last block has this method.

```{.python .input}
%%tab all
@add_to_class(A)
def do(self):
    print('class attribute "a" is', self.a)

a.do()
```

The second one is a utility class that saves all arguments in a class's `__init__` methods as class attributes.

```{.python .input}
%%tab all
class HyperParameters:  #@save
    def save_hyperparameters(self, ignore=[]):
        raise NotImplemented
```

We defer its implementation into :numref:`sec_utils`. To use it, we can have our class be a subclass of `HyperParameters` and call `save_hyperparameters` in the `__init__` method. For example

```{.python .input}
%%tab all
class B(d2l.HyperParameters):  # call the one saved in d2l with code implementation.
    def __init__(self, a, b, c):
        self.save_hyperparameters(ignore=['c'])
        print('self.a =', self.a, 'self.b =', self.b)
        print('there is no self.c =', not hasattr(self, 'c'))

B(a=1, b=2, c=3);
```

The last one is a class to plot points in animation. We will use it to show the training progress. Again, implementation is deferred to :numref:`sec_utils`. The draw function plots a point `(x, y)` in the figure, with `label` specific the legend. The optional `every_n` smooths the line by only showing $1/n$ points in the figure, whose values are averaged from the $n$ neighbor points in the original figure.

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
for x in np.arange(0, 10, 0.05):
    board.draw(x, np.sin(x), 'sin', every_n=5)
    board.draw(x, np.cos(x), 'cos', every_n=10)
```

## Model

The `Module` class  is the base class of all models we will implement. Minimally we need to define three methods. The `__init__` method stores the learnable parameters, the `training_step` method accepts a data batch to return the loss value, the `configure_optimizers` method returns the optimization method, or a list of them, that is used to update the learnable parameters. Optionally we can define `validation_step` to report the evaluation metrics.
Sometimes we put the code to compute the outputs into a separate `forward` method to make it more reusable.

```{.python .input}
%%tab all
class Module(d2l.nn_Module, d2l.HyperParameters):  #@save
    def __init__(self):
        super().__init__()
        self.board = ProgressBoard()
        if tab.selected('tensorflow'):
            self.training = None

    def loss(self, y_hat, y):
        raise NotImplementedError
        
    def forward(self, X):
        assert hasattr(self, 'net'), 'No neural network is defined'
        return self.net(X)
    
    if tab.selected('tensorflow'):
        def call(self, X, training=None):
            if training is not None:
                self.training = training
            return self.forward(X)

    def training_step(self, batch):
        X, y = batch
        l = self.loss(self(X), y)
        # Draw progress
        assert hasattr(self, 'trainer'), 'trainer is not inited'
        num_train = self.trainer.num_train_batches
        self.board.xlabel = 'epoch'
        self.board.draw(self.trainer.train_batch_idx / num_train, l, 
                        'train_loss', every_n=num_train // 5)
        return l

    def validation_step(self, batch):
        X, y = batch
        l = self.loss(self(X), y)
        # Draw progress
        self.board.draw(self.trainer.epoch+1, l, 'val_loss', 
                        every_n=self.trainer.num_val_batches)

    def configure_optimizers(self):
        raise NotImplementedError
```

:begin_tab:`mxnet`
You may notice that `Module` is a subclass of `nn.Block`, the base class of neural network in Gluon.
It provides convenient features to handle neural networks. For example, if we define a `forward` method, such as `forward(self, X)`, then for an instance `a` we can invoke this function by `a(X)`. In other words, it calls the `forward` method in the build-in `__call__` method. You can find more usages about `nn.Block` in :numref:`sec_model_construction`.
:end_tab:

:begin_tab:`pytorch`
You may notice that `Module` is a subclass of `nn.Module`, the base class of neural network in PyTorch.
It provides convenient features to handle neural networks. For example, if we define a `forward` method, such as `forward(self, X)`, then for an instance `a` we can invoke this function by `a(X)`. In other words, it calls the `forward` method in the build-in `__call__` method. You can find more usages about `nn.Block` in :numref:`sec_model_construction`.
:end_tab:

:begin_tab:`tensorflow`
You may notice that `Module` is a subclass of `tf.keras.Model`, the base class of neural network in TensorFlow.
It provides convenient features to handle neural networks. For example, it calls the `call` method in the build-in `__call__` method. Here we redirect `call` to the `forward` function, and saving its argument as a class attribute. We do it to make our code is more similar across different framework implementations.
:end_tab:

##  Data

The `DataModule` class is the base class for data. We often have the `__init__` function to prepare data, such as downloading and preprocessing the data. The `train_dataloader` returns the data loader for the training dataset. A data loader is a generator that yields a data batch each time. A data batch is then feed into the `training_step` method of `Module` to compute loss. There is an optional `val_dataloader` to return the validation dataset loader,
which yields data batches for the `validation_step` method in `Module`.

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

The `Trainer` class trains the learnable parameters in the `Module` class with data specified in `DataModule`. The key method is `fit`, which accepts two arguments: `model`, an instance of `Module`, and `data`, an instance of `DataModule`. It then iterate the data by `max_epochs` times to train the model. As before, we will defer the implementation of this function to later chapters.

```{.python .input}
%%tab all
class Trainer(d2l.HyperParameters):  #@save
    def __init__(self, max_epochs, num_gpus=0):
        self.save_hyperparameters()
        assert num_gpus == 0, 'Not support GPUs yet'

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

- Most code in this book will be organized into `Trainer` and subclasses of `Module` and `DataModule`.
