```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Concise Implementation of Linear Regression
:label:`sec_linear_concise`

Deep Learning has undergone a Cambrian Explosion of sorts over the past decade,
with a rapid proliferation of new techniques, applications, and algorithms. 
This is due to a fortuitous combination of multiple factors,
one of which is the powerful free tools
offered by a number of open source deep learning frameworks. 
Caffe, DistBelief and Theano arguably represent the
first generation of such models :cite:`jia2014caffe,dean2012large,bergstra2010theano` 
that found widespread adoption. 
In contrast to earlier (seminal) work such as SN2 (Simulateur Neuristique)
which provided a Lisp-like programming experience
:cite:`bottou1989cun` modern frameworks 
offer automatic differentiation
and the convenience of Python. 
Frameworks allow us to automate and modularize
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
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input  n=1}
%%tab pytorch
from d2l import torch as d2l
import numpy as np
import torch
from torch import nn
```

```{.python .input  n=1}
%%tab tensorflow
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

For standard operations, 
we can [**use a framework's predefined layers,**]
which allow us to focus
on the layers used to construct the model
rather than worrying about their implementation.
Recall the architecture of a single-layer network 
as described in :numref:`fig_single_neuron`.
The layer is called *fully connected*, 
since each of its inputs is connected 
to each of its outputs
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
In PyTorch, the fully connected layer is defined in the `Linear` class. 
Note that we passed two arguments into `nn.Linear`. 
The first specifies the input feature dimension (2), 
and the second specifies the output dimension (1).
:end_tab:

:begin_tab:`tensorflow`
In Keras, the fully connected layer is defined in the `Dense` class. 
Since we only want to generate a single scalar output, 
we set that number to 1. 
It is worth noting that, for convenience,
Keras does not require us to specify
the input shape for each layer.
We don't need to tell Keras
how many inputs go into this linear layer.
When we first try to pass data through our model,
e.g., when we execute `net(X)` later,
Keras will automatically infer 
the number of inputs to each layer.
We will describe how this works in more detail later.
:end_tab:

```{.python .input}
%%tab mxnet, tensorflow
class LinearRegression(d2l.Module):  #@save
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Dense(1)
            self.net.initialize()
        if tab.selected('tensorflow'):
            self.net = tf.keras.layers.Dense(1)
```

```{.python .input  n=2}
%%tab pytorch
class LinearRegression(d2l.Module):  #@save
    def __init__(self, num_inputs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Linear(num_inputs, 1)
```

In the forward method, we just evoke the built-in `__call__` function of the predefined layers to compute the outputs.

```{.python .input  n=3}
%%tab all
@d2l.add_to_class(LinearRegression)  #@save
def forward(self, X):
    """The linear regression model."""
    return self.net(X)
```

## Loss Function

:begin_tab:`mxnet`
The `loss` module defines many useful loss functions. 
For speed and convenience, we forgo implementing our own 
and choose the built-in `loss.L2Loss` instead. 
Because the `loss` that it returns is 
the squared error for each example, 
we use `mean`to average the loss across over the minibatch.
:end_tab:

:begin_tab:`pytorch`
[**The `MSELoss` class computes the mean squared error.**]
By default, `MSELoss` returns the average loss over examples. 
It is faster (and easier to use) than implementing our own.
:end_tab:

:begin_tab:`tensorflow`
The `MeanSquaredError` class computes the mean squared error.
By default it returns the average loss over examples.
:end_tab:

```{.python .input  n=3}
%%tab all
@d2l.add_to_class(LinearRegression)  #@save
def loss(self, y_hat, y):
    if tab.selected('mxnet'):
        fn = gluon.loss.L2Loss()
        return fn(y_hat, y).mean()
    if tab.selected('pytorch'):
        fn = nn.MSELoss()
        return fn(y_hat, y)
    if tab.selected('tensorflow'):
        fn = tf.keras.losses.MeanSquaredError()
        return fn(y, y_hat)
```

## Optimization Algorithm

:begin_tab:`mxnet`
Minibatch SGD is a standard tool
for optimizing neural networks
and thus Gluon supports it alongside a number of
variations on this algorithm through its `Trainer` class.
Note that Gluon's `Trainer` class stands 
for the optimization algorithm,
while the `Trainer` class we created in :numref:`sec_d2l_apis` 
contains the training function,
i.e., repeatedly call the optimizer
to update the model parameters.
When we instantiate `Trainer`,
we specify the parameters to optimize over,
obtainable from our model `net` via `net.collect_params()`,
the optimization algorithm we wish to use (`sgd`),
and a dictionary of hyperparameters
required by our optimization algorithm.

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
:end_tab:

:begin_tab:`tensorflow`
Minibatch SGD is a standard tool
for optimizing neural networks
and thus Keras supports it alongside a number of
variations on this algorithm in the `optimizers` module.
:end_tab:

```{.python .input  n=5}
%%tab all
@d2l.add_to_class(LinearRegression)  #@save
def configure_optimizers(self):
    if tab.selected('mxnet'):
        return gluon.Trainer(self.collect_params(),
                             'sgd', {'learning_rate': self.lr})
    if tab.selected('pytorch'):
        return torch.optim.SGD(self.parameters(), self.lr)
    if tab.selected('tensorflow'):
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
[**the training loop itself is the same
as the one we implemented from scratch.**]
So we just call the `fit` method 
(defined in :numref:`sec_linear_scratch`) 
to train our model.

```{.python .input}
%%tab all
if tab.selected('mxnet') or tab.selected('tensorflow'):
    model = LinearRegression(lr=0.03)
if tab.selected('pytorch'):
    model = LinearRegression(2, lr=0.03)
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)
```

Below, we 
[**compare the model parameters learned
by training on finite data
and the actual parameters**]
that generated our dataset.
To access parameters,
we first access the layer that we need from `net`
and then access its weights and bias.
As in our implementation from scratch,
note that our estimated parameters 
are close to their true counterparts.

```{.python .input}
%%tab all
@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self):
    if tab.selected('mxnet'):
        return (self.net.weight.data(), self.net.bias.data())
    if tab.selected('pytorch'):
        return (self.net.weight.data, self.net.bias.data)
    if tab.selected('tensorflow'):
        return (self.get_weights()[0], self.get_weights()[1])

w, b = model.get_w_b()
print(f'error in estimating w: {data.w - d2l.reshape(w, data.w.shape)}')
print(f'error in estimating b: {data.b - b}')
```

## Summary

This section contains the first 
implementation of a deep network (in this book)
to tap into the conveniences afforded 
by modern deep learning frameworks,
such as Gluon, JAX, Keras, PyTorch, and Tensorflow :cite:`abadi2016tensorflow,paszke2019pytorch,frostig2018compiling,chen2015mxnet`. 
We used framework defaults for loading data, defining a layer, 
a loss function, an optimizer and a training loop. 
Whenever the framework provides all necessary features,
it's generally a good idea to use them,
since the library implementations of these components 
tend to be heavily optimized for performance
and properly tested for reliability.
At the same time, try not to forget 
that these modules *can* be implemented directly.
This is especially important for aspiring researchers
who wish to live on the bleeding edge of model development,
where you will be inventing new components 
that cannot possibly exist in any current library.

:begin_tab:`mxnet`
In Gluon, the `data` module provides tools for data processing, 
the `nn` module defines a large number of neural network layers, 
and the `loss` module defines many common loss functions. 
Moreover, the `initializer` gives access 
to many choices for parameter initialization. 
Conveniently for the user,
dimensionality and storage are automatically inferred. 
A consequence of this lazy initialization is that 
you must not attempt to access parameters 
before they have been instantiated (and initialized).
:end_tab:

:begin_tab:`pytorch`
In PyTorch, the `data` module provides tools for data processing, 
the `nn` module defines a large number of neural network layers and common loss functions.
We can initialize the parameters by replacing their values 
with methods ending with `_`. 
Note that we need to specify the input dimensions of the network. 
While this is trivial for now, it can have significant knock-on effects 
when we want to design complex networks with many layers. 
Careful considerations of how to parametrize these networks 
is needed to allow portability.
:end_tab:

:begin_tab:`tensorflow`
In TensorFlow, the `data` module provides tools for data processing, 
the `keras` module defines a large number of neural network layers and common loss functions. 
Moreover, the `initializers` module provides various methods for model parameter initialization. 
Dimensionality and storage for networks are automatically inferred 
(but be careful not to attempt to access parameters before they have been initialized).
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
