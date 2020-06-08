# Custom Layers

One factor behind deep learning's success
is the availability of a wide range of layers
that can be composed in creative ways
to design architectures suitable
for a wide variety of tasks.
For instance, researchers have invented layers
specifically for handling images, text,
looping over sequential data,
performing dynamic programming, etc.
Sooner or later you will encounter (or invent)
a layer that does not exist yet in the framework,
In these cases, you must build a custom layer.
In this section, we show you how.

## Layers without Parameters

To start, we construct a custom layer (a block) 
that does not have any parameters of its own. 
This should look familiar if you recall our 
introduction to block in :numref:`sec_model_construction`. 
The following `CenteredLayer` class simply
subtracts the mean from its input. 
To build it, we simply need to inherit 
from the Block class and implement the `forward` method.

```{.python .input}
from mxnet import gluon, np, npx
from mxnet.gluon import nn
npx.set_np()

class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        return x - x.mean()
```


```{.python .input}
#@tab pytorch
import torch
from torch import nn

class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x - x.mean()
```

Let us verify that our layer works as intended by feeding some data through it.

```{.python .input}
layer = CenteredLayer()
layer(np.array([1, 2, 3, 4, 5]))
```


```{.python .input}
#@tab pytorch
layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))
```

We can now incorporate our layer as a component
in constructing more complex models.

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(128), CenteredLayer())
net.initialize()
```


```{.python .input}
#@tab pytorch
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
```

As an extra sanity check, we can send random data 
through the network and check that the mean is in fact 0.
Because we are dealing with floating point numbers, 
we may still see a *very* small nonzero number
due to quantization.

```{.python .input}
y = net(np.random.uniform(size=(4, 8)))
y.mean()
```


```{.python .input}
#@tab pytorch
y = net(torch.rand(4, 8))
y.mean()
```

## Layers with Parameters

Now that we know how to define simple layers,
let us move on to defining layers with parameters
that can be adjusted through training. 
To automate some of the routine work
the `Parameter` class 
provide some basic housekeeping functionality.
In particular, they govern access, initialization, 
sharing, saving, and loading model parameters. 
This way, among other benefits, we will not need to write
custom serialization routines for every custom layer.

:begin_tab:`mxnet`
The `Block` class contains a `params` variable
of the `ParameterDict` type. 
This dictionary maps strings representing parameter names
to model parameters (of the `Parameter` type). 
The `ParameterDict` also supplies a `get` function
that makes it easy to generate a new parameter
with a specified name and shape.
:end_tab:


```{.python .input}
params = gluon.ParameterDict()
params.get('param2', shape=(2, 3))
params
```


:begin_tab:`mxnet`
We now have all the basic ingredients that we need
to implement our own version of Gluon's `Dense` layer. 
Recall that this layer requires two parameters,
one to represent the weight and another for the bias. 
In this implementation, we bake in the ReLU activation as a default.
In the `__init__` function, `in_units` and `units`
denote the number of inputs and outputs, respectively.
:end_tab:

:begin_tab:`pytorch`
Now let's implement your own version of PyTorch's `Linear` layer. 
Recall that this layer requires two parameters,
one to represent the weight and another for the bias. 
In the `__init__` function, `in_units` and `units`
denote the number of inputs and outputs, respectively.
:end_tab:

```{.python .input}
class MyDense(nn.Block):
    # units: the number of outputs in this layer; in_units: the number of
    # inputs in this layer
    def __init__(self, units, in_units, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        linear = np.dot(x, self.weight.data(ctx=x.ctx)) + self.bias.data(ctx=x.ctx)
        return npx.relu(linear)
```


```{.python .input}
#@tab pytorch
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, x):
        return torch.matmul(x, self.weight.data) + self.bias.data
```

Next, we instantiate the `MyDense` class 
and access its model parameters.

```{.python .input}
dense = MyDense(units=3, in_units=5)
dense.params
```


```{.python .input}
#@tab pytorch
dense = MyLinear(5, 3)
dense.weight
```

We can directly carry out forward calculations using custom layers.

```{.python .input}
dense.initialize()
dense(np.random.uniform(size=(2, 5)))
```


```{.python .input}
#@tab pytorch
dense(torch.randn(2, 5))
```

We can also construct models using custom layers.
Once we have that we can use it just like the built-in dense layer.

```{.python .input}
net = nn.Sequential()
net.add(MyDense(8, in_units=64),
        MyDense(1, in_units=8))
net.initialize()
net(np.random.uniform(size=(2, 64)))
```


```{.python .input}
#@tab pytorch
net = nn.Sequential(MyLinear(64, 8), nn.ReLU(), MyLinear(8, 1))
net(torch.randn(2, 64))
```

## Summary

* We can design custom layers via the Block class. This allows us to define flexible new layers that behave differently from any existing layers in the library.
* Once defined, custom layers can be invoked in arbitrary contexts and architectures.
* Blocks can have local parameters, which are stored in a `ParameterDict` object in each Block's `params` attribute.


## Exercises

1. Design a layer that learns an affine transform of the data.
1. Design a layer that takes an input and computes a tensor reduction, 
   i.e., it returns $y_k = \sum_{i, j} W_{ijk} x_i x_j$.
1. Design a layer that returns the leading half of the Fourier coefficients of the data. 


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/58)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/59)
:end_tab:
