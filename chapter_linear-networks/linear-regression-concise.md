# Concise Implementation of Linear Regression
:label:`sec_linear_gluon`

Broad and intense interest in deep learning for the past several years
has inspired both companies, academics, and hobbyists
to develop a variety of mature open source frameworks
for automating the repetitive work of implementing
gradient-based learning algorithms.
In the previous section, we relied only on
(i) tensors for data storage and linear algebra;
and (ii) auto differentiation for calculating derivatives.
In practice, because data iterators, loss functions, optimizers,
and neural network layers (and some whole architectures)
are so common, modern libraries implement these components for us as well.

In this section, we will show you how to implement
the linear regression model from :numref:`sec_linear_scratch`
concisely by using framework's high-level APIs.

## Generating the Dataset

To start, we will generate the same dataset as in the previous section.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
npx.set_np()

true_w = np.array([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import numpy as np
import torch
from torch.utils import data

true_w = torch.Tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
labels = labels.reshape(-1,1)
```

## Reading the Dataset

Rather than rolling our own iterator,
we can call upon the `data` module to read data.
The first step will be to instantiate an `ArrayDataset`.
This object's constructor takes one or more tensors as arguments.
Here, we pass in `features` and `labels` as arguments.
Next, we will use the `ArrayDataset` to instantiate a `DataLoader`,
which also requires that we specify a `batch_size`
and specify a Boolean value `shuffle` indicating whether or not
we want the `DataLoader` to shuffle the data
on each epoch (pass through the dataset).

```{.python .input}
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a Gluon data loader."""
    dataset = gluon.data.ArrayDataset(*data_arrays)
    return gluon.data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
```

```{.python .input}
#@tab pytorch
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a PyTorch data loader"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
```

Now we can use `data_iter` in much the same way as we called
the `data_iter` function in the previous section.
To verify that it is working, we can read and print
the first minibatch of instances. Comparing to :numref:`sec_linear_scratch`, here we use `iter` to construct an Python iterator and then use `next` to obtain the first item from the iterator.

```{.python .input}
next(iter(data_iter))
```

```{.python .input}
#@tab pytorch
next(iter(data_iter))
```

## Defining the Model

When we implemented linear regression from scratch
(in :numref:`sec_linear_scratch`),
we defined our model parameters explicitly
and coded up the calculations to produce output
using basic linear algebra operations.
You *should* know how to do this.
But once your models get more complex,
and once you have to do this nearly every day,
you will be glad for the assistance.
The situation is similar to coding up your own blog from scratch.
Doing it once or twice is rewarding and instructive,
but you would be a lousy web developer
if every time you needed a blog you spent a month
reinventing the wheel.

For standard operations, we can use the framework's predefined layers,
which allow us to focus especially
on the layers used to construct the model
rather than having to focus on the implementation.
To define a linear model, we first import the `nn` module,
which defines a large number of neural network layers
(note that "nn" is an abbreviation for neural networks).
We will first define a model variable `net`,
which will refer to an instance of the `Sequential` class.
The `Sequential` class defines a container
for several layers that will be chained together.
Given input data, a `Sequential` passes it through
the first layer, in turn passing the output
as the second layer's input and so forth.
In the following example, our model consists of only one layer,
so we do not really need `Sequential`.
But since nearly all of our future models
will involve multiple layers,
we will use it anyway just to familiarize you
with the most standard workflow.

Recall the architecture of a single-layer network as shown in :numref:`fig_singleneuron`.
The layer is said to be *fully-connected*
because each of its inputs are connected to each of its outputs
by means of a matrix-vector multiplication.

![Linear regression is a single-layer neural network. ](../img/singleneuron.svg)
:label:`fig_singleneuron`

```{.python .input}
from mxnet.gluon import nn
net = nn.Sequential()
net.add(nn.Dense(1))
```

```{.python .input}
#@tab pytorch
from torch import nn
net = nn.Sequential(nn.Linear(2, 1))
```

:begin_tab:`mxnet`
In Gluon, the fully-connected layer is defined in the `Dense` class.
Since we only want to generate a single scalar output,
we set that number to $1$.

It is worth noting that, for convenience,
Gluon does not require us to specify
the input shape for each layer.
So here, we do not need to tell Gluon
how many inputs go into this linear layer.
When we first try to pass data through our model,
e.g., when we execute `net(X)` later,
Gluon will automatically infer the number of inputs to each layer.
We will describe how this works in more detail
in the chapter "Deep Learning Computation".
:end_tab:

:begin_tab:`pytorch`
In Gluon, the fully-connected layer is defined in the `Linear` class. Note that we passed two arguments into `nn.Linear`. The first one specifies the input feature dimension, which is 2, and the second one is the output feature dimension, which is a single scalar and therefore 1.
:end_tab:

## Initializing Model Parameters

Before using `net`, we need to initialize the model parameters,
such as the weights and biases in the linear regression model.

:begin_tab:`mxnet`
We will import the `initializer` module from MXNet.
This module provides various methods for model parameter initialization.
Gluon makes `init` available as a shortcut (abbreviation)
to access the `initializer` package.
By calling `init.Normal(sigma=0.01)`,
we specify that each *weight* parameter
should be randomly sampled from a normal distribution
with mean $0$ and standard deviation $0.01$.
The *bias* parameter will be initialized to zero by default.
Both the weight vector and bias will have attached gradients.
:end_tab:

:begin_tab:`pytorch`
As we have specified the input and output dimensions when constructing `nn.Linear`. Now we access the parameters directly to specify there initial values. We first locate the layer by `net[0]`, which is the first layer in the network, and then use the `weight.data` and `bias.data` methods to access the parameters. Next we use the replace methods `uniform_` and `fill_` to overwrite parameter values.
:end_tab:

```{.python .input}
from mxnet import init
net.initialize(init.Normal(sigma=0.01))
```

```{.python .input}
#@tab pytorch
net[0].weight.data.uniform_(0.0, 0.01)
net[0].bias.data.fill_(0)
```

:begin_tab:`mxnet`
The code above may look straightforward but you should note
that something strange is happening here.
We are initializing parameters for a network
even though Gluon does not yet know
how many dimensions the input will have!
It might be $2$ as in our example or it might be $2000$.
Gluon lets us get away with this because behind the scenes,
the initialization is actually *deferred*.
The real initialization will take place only
when we for the first time attempt to pass data through the network.
Just be careful to remember that since the parameters
have not been initialized yet,
we cannot access or manipulate them.
:end_tab:

:begin_tab:`pytorch`

:end_tab:

## Defining the Loss Function

:begin_tab:`mxnet`
In Gluon, the `loss` module defines various loss functions.
We will use the imported module `loss` with the pseudonym `gloss`
to avoid confusing it for the variable
holding our chosen loss function.
In this example, we will use the Gluon
implementation of squared loss (`L2Loss`).
:end_tab:

:begin_tab:`pytorch`
The `MSELoss` class compute the mean squared error, also known as squared L2 norm. In default it returns the averaged loss over examples.
:end_tab:

```{.python .input}
from mxnet.gluon import loss as gloss
loss = gloss.L2Loss()
```

```{.python .input}
#@tab pytorch
loss = nn.MSELoss()
```

## Defining the Optimization Algorithm

:begin_tab:`mxnet`
Minibatch SGD and related variants
are standard tools for optimizing neural networks
and thus Gluon supports SGD alongside a number of
variations on this algorithm through its `Trainer` class.
When we instantiate the `Trainer`,
we will specify the parameters to optimize over
(obtainable from our net via `net.collect_params()`),
the optimization algorithm we wish to use (`sgd`),
and a dictionary of hyper-parameters
required by our optimization algorithm.
SGD just requires that we set the value `learning_rate`,
(here we set it to 0.03).
:end_tab:

:begin_tab:`pytorch`
Minibatch SGD and related variants
are standard tools for optimizing neural networks
and thus PyTorch supports SGD alongside a number of
variations on this algorithm in the `optim` module.
When we instantiate a SGD instance,
we will specify the parameters to optimize over
(obtainable from our net via `net.parameters()`), with a dictionary of hyper-parameters
required by our optimization algorithm.
SGD just requires that we set the value `learning_rate`,
(here we set it to 0.03).
:end_tab:

```{.python .input}
from mxnet import gluon
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD(net.parameters(), lr = .03)
```

## Training

You might have noticed that expressing our model through Gluon
requires comparatively few lines of code.
We did not have to individually allocate parameters,
define our loss function, or implement stochastic gradient descent.
Once we start working with much more complex models,
Gluon's advantages will grow considerably.
However, once we have all the basic pieces in place,
the training loop itself is strikingly similar
to what we did when implementing everything from scratch.

To refresh your memory: for some number of epochs,
we will make a complete pass over the dataset (train_data),
iteratively grabbing one minibatch of inputs
and the corresponding ground-truth labels.
For each minibatch, we go through the following ritual:

* Generate predictions by calling `net(X)` and calculate the loss `l` (the forward pass).
* Calculate gradients by calling `l.backward()` (the backward pass).
* Update the model parameters by invoking our SGD optimizer (note that `trainer` already knows which parameters to optimize over, so we just need to pass in the minibatch size.

For good measure, we compute the loss after each epoch and print it to monitor progress.

```{.python .input}
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))
```

```{.python .input}
#@tab pytorch
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print('epoch {}, loss {}'.format(epoch, l))
```

Below, we compare the model parameters learned by training on finite data
and the actual parameters that generated our dataset.
To access parameters with Gluon,
we first access the layer that we need from `net`
and then access that layer's weight (`weight`) and bias (`bias`).
To access each parameter's values as a tensor,
we invoke its `data` method.
As in our from-scratch implementation,
note that our estimated parameters are
close to their ground truth counterparts.

```{.python .input}
w = net[0].weight.data()
print('Error in estimating w', true_w.reshape(w.shape) - w)
b = net[0].bias.data()
print('Error in estimating b', true_b - b)
```

```{.python .input}
#@tab pytorch
w = net[0].weight.data
print('Error in estimating w', true_w.reshape(w.shape) - w)
b = net[0].bias.data
print('Error in estimating b', true_b - b)
```

## Summary

:begin_tab:`mxnet`
* Using Gluon, we can implement models much more succinctly.
* In Gluon, the `data` module provides tools for data processing, the `nn` module defines a large number of neural network layers, and the `loss` module defines many common loss functions.
* MXNet's module `initializer` provides various methods for model parameter initialization.
* Dimensionality and storage are automatically inferred (but be careful not to attempt to access parameters before they have been initialized).
:end_tab:

:begin_tab:`pytorch`
* Using PyTorch's high-level APIs, we can implement models much more succinctly.
* In PyTorch, the `data` module provides tools for data processing, the `nn` module defines a large number of neural network layers and common loss functions.
* We can initialize the parameters by replacing their values with methods ending with `_`.
:end_tab:

## Exercises

:begin_tab:`mxnet`
1. If we replace `l = loss(output, y)` with `l = loss(output, y).mean()`, we need to change `trainer.step(batch_size)` to `trainer.step(1)` for the code to behave identically. Why?
1. Review the MXNet documentation to see what loss functions and initialization methods are provided in the modules `gluon.loss` and `init`. Replace the loss by Huber's loss.
1. How do you access the gradient of `dense.weight`?

[Discussions](https://discuss.d2l.ai/t/44)
:end_tab:

:begin_tab:`pytorch`
1. If we replace `nn.MSELoss(reduction='sum')` with `nn.MSELoss()`, how can we change the learning rate for the code to behave identically. Why?
1. Review the PyTorch documentation to see what loss functions and initialization methods are provided. Replace the loss by Huber's loss.
1. How do you access the gradient of `net[0].weight`?

[Discussions](https://discuss.d2l.ai/t/45)
:end_tab:
