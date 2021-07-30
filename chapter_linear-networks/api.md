# The D2L APIs

The linear regression model is one of the simplest machine learning models. 
It, however, shares the same elements with others models we will discuss in this book.
Before diving into the code implementations of the linear regression model, 
let's first explain how we organize our code through this book. 
It will make you easier to read our code and even use it in your projects.

Through this book we will use minibatch stochastic gradient decent to train our models. 
It opens the door for us to adopt a consistent API design. In particular, we organize our code
into three parts: data, model, and training. 


```{.python .input}
import time
from d2l import mxnet as d2l
from mxnet.gluon import nn
```

```{.python .input}
#@tab pytorch
import time
from d2l import torch as d2l
import torch
from torch import nn

```

## Utilities

Let's first introduce utility functions and classes. We will adopt the object-oriented programming that is common for Python libraries. It's, however, used less in notebooks where we often introduce keep a code block short for readability. The first one is function that allows us to register a function to a class so we can split class methods into multiple code blocks. 

```{.python .input}
#@tab all
def add_to_class(Class):  #@save
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper
```

For example, if we plan to implement a class `A` with a method `do`. Instead of having code for both `A` and `do` in the same code block, we can first declare the class `A` and construct an instance `a`. 

```{.python .input}
#@tab all
class A:
    def __init__(self):
        self.a = 1

a = A()
```

Next we define the class method `do` as we do normally but not in the class `A`'s scope. Instead, we decorate this function by `add_to_class` with class `A` as its argument. Then we can see that the instance `a` we created in the last block has this method. 

```{.python .input}
#@tab all
@add_to_class(A)
def do(self):
    print('class attribute `a` is', self.a)

a.do()
```

The second one allows us to save all arguments in a class's `__init__` methods as class attributes.

```{.python .input}
#@tab all
class HyperParameters:  #@save
    def save_hyperparameters(self, ignore=[]):
        raise NotImplemented
```

We defer its implementation into :numref:`sec_utils`. To use it, we can have our class be a subclass of `HyperParameters` and call `save_hyperparameters` in the `__init__` method. For example

```{.python .input}
#@tab all
class B(d2l.HyperParameters):  # call the one saved in d2l with code implementation.
    def __init__(self, a, b, c):
        self.save_hyperparameters(ignore=['c'])
        print('self.a =', self.a, 'self.b =', self.b)
        print('there is no self.c =', not hasattr(self, 'c'))

B(a=1, b=2, c=3);
```

The third one is a plot board with animation. Again, implementation is deferred to :numref:`sec_utils`

```{.python .input}
#@tab all
class ProgressBoard(d2l.HyperParameters):  #@save
    """Plot data points in animation."""
    def __init__(self, x, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5)):
        self.save_hyperparameters()
        self.points = []
        self.lines = {}

    def draw(self, points, every_n=1):
        raise NotImplemented
```

Take an example. The `every_n` argument draw a point in the plot for every $n$ points passed to `draw`. The draw point is the averaged value. 

```{.python .input}
#@tab all
points = [{'x':1, 'y1':2, 'y2':3}, {'x':2, 'y1':3, 'y2':4},
          {'x':3, 'y1':2, 'y2':3}, {'x':4, 'y1':3, 'y2':4}]
board = d2l.ProgressBoard(x='x')
for p in points:
    board.draw(p)
    time.sleep(1)
```

## Model

In the model part, we define the neural network architecture to predict on a data batch, specify how to compute the loss and evaluation metrics, and picking the optimization method. The following class `Module` is the base class of all models we will implement. Minimally we need to define three methods. The `__init__` method stores the learnable parameters, the `training_step` method accepts a data batch, with its index $i=0,1,\ldots$ to return the loss value. 
We will often put the code to compute the outputs into a separate `forward` method so it can be reusable later. 
The `configure_optimizers` method returns the optimization method, or a list of them, that is used to update the learnable parameters. Optionally we can define `validation_step` to report the evaluation metrics. 

TODO, explain why inherent the `nn` class

```{.python .input}
#@tab all
class Module(d2l.nn_Module, d2l.HyperParameters):  #@save
    def __init__(self):
        super().__init__()
        self.board = ProgressBoard(x='step')

    def training_step(self, batch, batch_idx):
        raise NotImplemented

    def validaton_step(self):
        return None

    def configure_optimizers(self):
        raise NotImplemented
```

##  Data

```{.python .input}
#@tab all
class DataModule(d2l.HyperParameters):  #@save
    def __init__(self, root='../data', num_workers=4):
        self.save_hyperparameters()

    def train_dataloader(self):
        """Returns the dataloader for the training dataset."""
        raise NotImplemented

    def val_dataloader(self):
        """Returns the dataloader for the validation dataset."""
```

## Training API

We can construct a trainer by specifying the maximal number of epochs. Our fully functional trainer allows other options such as the number of GPUs, here we keep it simple and will discuss it later.

The `fit` method accepts two arguments: `model`, an instance of `Module`, and `data`, an instance of `DataModule`. 

```{.python .input}
#@tab all
class Trainer(d2l.HyperParameters):  #@save
    def __init__(self, max_epochs):
        self.save_hyperparameters()

    def fit(self, model, data):
        raise NotImplemented
```
