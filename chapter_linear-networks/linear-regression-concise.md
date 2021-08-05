# Concise Implementation of Linear Regression
:label:`sec_linear_concise`

Deep Learning has witnessed a Cambrian Explosion of sorts over the past decade.
The sheer number of techniques, applications and algorithms by far surpasses the
progress of previous decades. This is due to a fortuitous combination of multiple
factors, not the least due to the ease of implementation offered by a number
of open source deep learning frameworks. Caffe, DistBelief and Theano arguably represent the
first generation of such models :cite:`jia2014caffe,dean2012large,bergstra2010theano` that
widespread adoption. In contrast to earlier (seminal) work such as SN2 (Simulateur Neuristique)
which provided a Lisp-like programming experience
:cite:`bottou1989cun` modern frameworks offer automatic differentiation and the convenience
of Python. Frameworks allow us to automate and modularize
the repetitive work of implementing gradient-based learning algorithms.

In :numref:`sec_linear_scratch`, we relied only on
(i) tensors for data storage and linear algebra;
and (ii) auto differentiation for calculating gradients.
In practice, because data iterators, loss functions, optimizers,
and neural network layers
are so common, modern libraries implement these components for us as well.
In this section, (**we will show you how to implement
the linear regression model**) from :numref:`sec_linear_scratch`
(**concisely by using high-level APIs**) of deep learning frameworks.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import numpy as np
import torch
```

```{.python .input  n=1}
#@tab tensorflow
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

## Defining the Model

When we implemented linear regression from scratch
in :numref:`sec_linear_scratch`,
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
if you spent a month reinventing the wheel.

For standard operations, we can [**use a framework's predefined layers,**]
which allow us to focus
on the layers used to construct the model
rather than worrying about their implementation.
We will first define a model `net`,
which will refer to an instance of the `Sequential` class.
The `Sequential` class defines a container
for several layers that will be chained together.
Given input data, a `Sequential` instance passes it through
the first layer, in turn passing the output
as the second layer's input and so forth.
For now our model consists of only one layer (we are implementing a linear model after all).
As such, we do not really need `Sequential`.
But since nearly all of our future models
will involve multiple layers,
we use it anyway just to familiarize ourselves
with the workflow.

Recall the architecture of a single-layer network as described in :numref:`fig_single_neuron`.
The layer is called *fully connected*, since each of its inputs is connected to each of its outputs
by means of a matrix-vector multiplication.

:begin_tab:`mxnet`
In Gluon, the fully connected layer is defined in the `Dense` class.
Since we only want to generate a single scalar output,
we set that number to 1.
It is worth noting that, for convenience,
Gluon does not require us to specify
the input shape for each layer.
Hence we don't need to tell Gluon
how many inputs go into this linear layer.
When we first pass data through our model,
e.g., when we execute `net(X)` later,
Gluon will automatically infer the number of inputs to each layer and
thus instantiate the correct model.
We will describe how this works in more detail later.
:end_tab:

:begin_tab:`pytorch`
In PyTorch, the fully connected layer is defined in the `Linear` class. Note that we passed two arguments into `nn.Linear`. The first one specifies the input feature dimension, which is 2, and the second one is the output feature dimension, which is a single scalar and therefore 1.
:end_tab:

:begin_tab:`tensorflow`
In Keras, the fully connected layer is defined in the `Dense` class. Since we only want to generate a single scalar output, we set that number to 1.

It is worth noting that, for convenience,
Keras does not require us to specify
the input shape for each layer.
We don't need to tell Keras
how many inputs go into this linear layer.
When we first try to pass data through our model,
e.g., when we execute `net(X)` later,
Keras will automatically infer the number of inputs to each layer.
We will describe how this works in more detail later.
:end_tab:

```{.python .input}
# `nn` is an abbreviation for neural networks
from mxnet.gluon import nn

class LinearRegression(d2l.Module):  #@save
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Dense(1)
        self.net.initialize()
```

```{.python .input}
#@tab pytorch
# `nn` is an abbreviation for neural networks
from torch import nn

class LinearRegression(d2l.Module):  #@save
    def __init__(self, num_inputs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Linear(num_inputs, 1)
```

```{.python .input  n=2}
#@tab tensorflow
# `keras` is the high-level API for TensorFlow

class LinearRegression(d2l.Module):  #@save
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = tf.keras.layers.Dense(1)
```

## Loss Function

:begin_tab:`mxnet`
The `loss` module defines many useful loss functions. For speed and convenience
we forgo implementing our own and choose the built-in `loss.L2Loss` instead. $L_2$ loss is a bit
of a misnomer, albeit a pervasive one in deep learning, since we are simply computing the mean squared error, i.e., the squared distance between estimates and labels. It has very little to do with the space of Lebesque-measurable functions, denoted by $L_2$ in mathematics.
:end_tab:

:begin_tab:`pytorch`
[**The `MSELoss` class computes the mean squared error.**]
By default it returns the average loss over examples. It is faster (and easier to use) than implementing our own.
:end_tab:

:begin_tab:`tensorflow`
The `MeanSquaredError` class computes the mean squared error.
By default it returns the average loss over examples.
:end_tab:

```{.python .input}
#@tab mxnet, pytorch
@d2l.add_to_class(LinearRegression)
def forward(self, X):
    """The linear regression model."""
    return self.net(X)
```

```{.python .input}
@d2l.add_to_class(LinearRegression)
def training_step(self, batch):
    X, y = batch
    loss = gluon.loss.L2Loss()
    l = loss(self(X), y).mean()
    self.board.xlabel = 'step'
    self.board.draw(self.trainer.train_batch_idx, l, 'loss', every_n=10)
    return l
```

```{.python .input}
#@tab pytorch
@d2l.add_to_class(LinearRegression)
def training_step(self, batch):
    X, y = batch
    loss = nn.MSELoss()
    l = loss(self(X), y)
    epoch = self.trainer.train_batch_idx / self.trainer.num_train_batches
    self.board.xlabel = 'epoch'
    self.board.draw(epoch, l, 'train_loss', every_n=50)
    return l
```

```{.python .input  n=3}
#@tab tensorflow
@d2l.add_to_class(LinearRegression)
def forward(self, X):
    """The linear regression model."""
    return self.net(X)

@d2l.add_to_class(LinearRegression)
def training_step(self, batch):
    X, y = batch
    loss = tf.keras.losses.MeanSquaredError()
    l = loss(self(X), y)
    self.board.xlabel = 'step'
    self.board.draw(self.trainer.train_batch_idx, l, 'loss', every_n=10)
    return l
```

## Optimization Algorithm

:begin_tab:`mxnet`
Minibatch SGD is a standard tool
for optimizing neural networks
and thus Gluon supports it alongside a number of
variations on this algorithm through its `Trainer` class.
When we instantiate `Trainer`,
we specify the parameters to optimize over,
obtainable from our model `net` via `net.collect_params()`,
the optimization algorithm we wish to use (`sgd`),
and a dictionary of hyperparameters
required by our optimization algorithm.
Minibatch stochastic gradient descent just requires that
we set the value `learning_rate`. We use 0.03.
:end_tab:

:begin_tab:`pytorch`
Minibatch SGD is a standard tool
for optimizing neural networks
and thus PyTorch supports it alongside a number of
variations on this algorithm in the `optim` module.
When we (**instantiate an `SGD` instance,**)
we specify the parameters to optimize over,
obtainable from our net via `net.parameters()`,
with a dictionary of hyperparameters
required by our optimization algorithm.
Minibatch SGD just requires that
we set the learning rate `lr`. We use 0.03.
:end_tab:

:begin_tab:`tensorflow`
Minibatch SGD is a standard tool
for optimizing neural networks
and thus Keras supports it alongside a number of
variations on this algorithm in the `optimizers` module.
Minibatch SGD just requires that
we set the value `learning_rate`. We use 0.03.
:end_tab:

```{.python .input}
@d2l.add_to_class(LinearRegression)
def configure_optimizers(self):
    return gluon.Trainer(self.collect_params(),
                         'sgd', {'learning_rate': self.lr})
```

```{.python .input}
#@tab pytorch
@d2l.add_to_class(LinearRegression)
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), self.lr)
```

```{.python .input}
#@tab tensorflow
@d2l.add_to_class(LinearRegression)
def configure_optimizers(self):
    return tf.keras.optimizers.SGD(self.lr)
```

## Training

You might have noticed that expressing our model through
high-level APIs of a deep learning framework
requires comparatively few lines of code.
We did not have to allocate parameters individually,
define our loss function, or implement minibatch SGD.
Once we start working with much more complex models,
the advantages of the high-level API will grow considerably.
Now that we have all the basic pieces in place,
[**the training loop itself is strikingly similar
to the one we obtained by implementing everything from scratch.**]

For some number of epochs,
we will make a pass over the dataset (`train_data`),
iteratively grabbing a minibatch of inputs
and labels at a time. For each minibatch, the `fit` function of the trainer goes
through the following steps:

* Generate predictions by calling `net(X)` and calculate the loss `l` (the forward propagation).
* Calculate gradients by running the backpropagation.
* Update the model parameters by invoking our optimizer.

For good measure, it also computes the loss after each epoch and print it to monitor progress.

```{.python .input}
#@tab pytorch
model = LinearRegression(2, lr=0.03)
```

```{.python .input}
#@tab mxnet, tensorflow
model = LinearRegression(lr=0.03)
```

```{.python .input}
#@tab all
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)
```

As we can see, the algorithm converges very quickly, yielding an even lower loss
than before, even though we implemented the same model. The solution for this mystery
can be found in the SGD solver. As we will see in :ref:`chap_optimization`, the SGD solver
doesn't implement a vanilla Stochastic Gradient Descent algorithm but adds a number of improvements to
make it more robust and accelerate convergence.

Below, we [**compare the model parameters learned by training on finite data
and the actual parameters**] that generated our dataset.
To access parameters,
we first access the layer that we need from `net`
and then access its weights and bias.
As in our implementation from-scratch,
note that our estimated parameters are
close to their true counterparts.

```{.python .input}
w = model.net.weight.data()
b = model.net.bias.data()
```

```{.python .input}
#@tab pytorch
w = model.net.weight.data
b = model.net.bias.data
```

```{.python .input}
#@tab tensorflow
w = model.get_weights()[0]
b = model.get_weights()[1]
```

```{.python .input}
#@tab all
print(f'error in estimating w: {data.w - d2l.reshape(w, data.w.shape)}')
print(f'error in estimating b: {data.b - b}')
```

## Summary

This section contains the first 'modern' implementation of a deep network that we encounter. By modern, we
mean an implementation that uses many of the conveniences afforded by a modern deep learning framework, such as Gluon, JAX, Keras, PyTorch, or Tensorflow :cite:`abadi2016tensorflow,paszke2019pytorch,frostig2018compiling,chen2015mxnet`. More to the point, we used framework-defaults for loading data, defining a layer, a loss function, an optimizer and a training loop. Whenever the framework provides all necessary features, this is the recommended way to proceed, since these components are typically heavily optimized. At the same time, we urge you not to forget that these modules *can* be implemented directly. This matters particularly for researchers at the bleeding edge of model development where not all components for a new model will not exist already in the researcher's Lego toolkit.

:begin_tab:`mxnet`
In Gluon, the `data` module provides tools for data processing, the `nn` module defines a large number of neural network layers, and the `loss` module defines many common loss functions. Moreover, the `initializer` gives access
to many choices for parameter initialization. Conveniently for the user,
dimensionality and storage are automatically inferred. A consequence of this lazy initialization is that you must not attempt to access parameters before they have been instantiated (and initialized).
:end_tab:

:begin_tab:`pytorch`
In PyTorch, the `data` module provides tools for data processing, the `nn` module defines a large number of neural network layers and common loss functions. We can initialize the parameters by replacing their values with methods ending with `_`. Note that we need to specify the input dimensions of the network. While this is trivial for now, it can have significant knock-on effects when we want to design complex networks with many layers. Careful considerations of how to parametrize these networks is needed to allow portability.
:end_tab:

:begin_tab:`tensorflow`
In TensorFlow, the `data` module provides tools for data processing, the `keras` module defines a large number of neural network layers and common loss functions. Moreover, the `initializers` module provides various methods for model parameter initialization. Dimensionality and storage for networks are automatically inferred (but be careful not to attempt to access parameters before they have been initialized).
:end_tab:

## Exercises

1. How would you need to change the learning rate if you replace the aggregate loss over the minibatch
   with an average over the loss on the minibatch?
1. Review the framework documentation to see which loss functions are provided. In particular,
   replace the squared loss with Huber's robust loss function. That is, use the loss function
   $$l(y,y') = \begin{cases}|y-y'| -\frac{\sigma}{2} & \text{ if } |y-y'| > \sigma \\ \frac{1}{2 \sigma} (y-y')^2 & \text{ otherwise}\end{cases}$$
1. How do you access the gradient of the linear part of the model, i.e.,
1. How does the solution change if you change the learning rate and the number of epochs? Does it keep on improving?
1. How does the solution change as you change the amount of data generated. $\sin x$
    1. Plot the estimation error for $\hat{\mathbf{w}} - \mathbf{w}$ and $\hat{b} - b$ as a function of the amount of data. Hint: increase the amount of data logarithmically rather than linearly, i.e. 5, 10, 20, 50 ... 10,000 rather than 1,000, 2,000 ... 10,000.
    2. Why is the suggestion in the hint appropriate?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/44)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/45)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/204)
:end_tab:
