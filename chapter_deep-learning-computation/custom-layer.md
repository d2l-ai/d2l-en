# Custom Layers

One of factors behind deep learnings success
is the availability of a wide range of layers
that can be composed in creative ways
to design architectures suitable
for a wide variety of tasks.
For instance, researchers have invented layers
specifically for handling images, text,
looping over sequential data,
performing dynamic programming, etc.
Sooner or later you will encounter (or invent)
a layer that does not exist yet in Gluon,
In these cases, you must build a custom layer.
In this section, we show you how.

## Layers without Parameters

To start, we construct a custom layer (a Block) 
that does not have any parameters of its own. 
This should look familiar if you recall our 
introduction to Gluon's `Block` in :numref:`sec_model_construction`. 
The following `CenteredLayer` class simply
subtracts the mean from its input. 
To build it, we simply need to inherit 
from the Block class and implement the `forward` method.

```{.python .input  n=1}
from mxnet import gluon, np, npx
from mxnet.gluon import nn
npx.set_np()

class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x - x.mean()
```

Let us verify that our layer works as intended by feeding some data through it.

```{.python .input  n=2}
layer = CenteredLayer()
layer(np.array([1, 2, 3, 4, 5]))
```

We can now incorporate our layer as a component
in constructing more complex models.

```{.python .input  n=3}
net = nn.Sequential()
net.add(nn.Dense(128), CenteredLayer())
net.initialize()
```

As an extra sanity check, we can send random data 
through the network and check that the mean is in fact 0.
Because we are dealing with floating point numbers, 
we may still see a *very* small nonzero number
due to quantization.

```{.python .input  n=4}
y = net(np.random.uniform(size=(4, 8)))
y.mean()
```

## Layers with Parameters

Now that we know how to define simple layers
let us move on to defining layers with parameters
that can be adjusted through training. 
To automate some of the routine work
the `Parameter` class and the `ParameterDict` dictionary 
provide some basic housekeeping functionality.
In particular, they govern access, initialization, 
sharing, saving and loading model parameters. 
This way, among other benefits, we will not need to write
custom serialization routines for every custom layer.

The `Block` class contains a `params` variable
of the `ParameterDict` type. 
This dictionary maps strings representing parameter names
to model parameters (of the `Parameter` type). 
The `ParameterDict` also supplied a `get` function
that makes it easy to generate a new parameter
with a specified name and shape.

```{.python .input  n=7}
params = gluon.ParameterDict()
params.get('param2', shape=(2, 3))
params
```

We now have all the basic ingredients that we need
to implement our own version of Gluon's `Dense` layer. 
Recall that this layer requires two parameters,
one to represent the weight and another for the bias. 
In this implementation, we bake in the ReLU activation as a default.
In the `__init__` function, `in_units` and `units`
denote the number of inputs and outputs, respectively.

```{.python .input  n=19}
class MyDense(nn.Block):
    # units: the number of outputs in this layer; in_units: the number of
    # inputs in this layer
    def __init__(self, units, in_units, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        linear = np.dot(x, self.weight.data()) + self.bias.data()
        return npx.relu(linear)
```

Naming our parameters allows us to access them 
by name through dictionary lookup later.
Generally, you will want to give your variables
simple names that make their purpose clear.
Next, we instantiate the `MyDense` class 
and access its model parameters.
Note that the Block's name is automatically
prepended to each Parameter's name.

```{.python .input}
dense = MyDense(units=3, in_units=5)
dense.params
```

We can directly carry out forward calculations using custom layers.

```{.python .input  n=20}
dense.initialize()
dense(np.random.uniform(size=(2, 5)))
```

We can also construct models using custom layers.
Once we have that we can use it just like the built-in dense layer.
The only exception is that in our case,
shape inference is not automatic. 
If you are interested in these bells and whisteles,
please consult the [MXNet documentation](http://www.mxnet.io)
for details on how to implement shape inference in custom layers.

```{.python .input  n=19}
net = nn.Sequential()
net.add(MyDense(8, in_units=64),
        MyDense(1, in_units=8))
net.initialize()
net(np.random.uniform(size=(2, 64)))
```

## Summary

* We can design custom layers via the Block class. This allows us to define flexible new layers that behave differently from any existing layers in the library.
* Once defined, custom layers can be invoked in arbitrary contexts and architectures.
* Blocks can have local parameters, which are stored as a `ParameterDict` object in each Blovk's `params` attribute.


## Exercises

1. Design a layer that learns an affine transform of the data.
1. Design a layer that takes an input and computes a tensor reduction, 
   i.e., it returns $y_k = \sum_{i, j} W_{ijk} x_i x_j$.
1. Design a layer that returns the leading half of the Fourier coefficients of the data. Hint: look up the `fft` function in MXNet.

## [Discussions](https://discuss.mxnet.io/t/2328)

![](../img/qr_custom-layer.svg)
