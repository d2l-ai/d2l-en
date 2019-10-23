# Custom Layers

One of the reasons for the success of deep learning can be found in the wide range of layers that can be used in a deep network. This allows for a tremendous degree of customization and adaptation. For instance, scientists have invented layers for images, text, pooling, loops, dynamic programming, even for computer programs. Sooner or later you will encounter a layer that does not exist yet in Gluon, or even better, you will eventually invent a new layer that works well for your problem at hand. This is when it is time to build a custom layer. This section shows you how.

## Layers without Parameters

Since this is slightly intricate, we start with a custom layer (aka Block) that
does not have any inherent parameters. Our first step is very similar to when we
introduced blocks in :numref:`sec_model_construction`. The following
`CenteredLayer` class constructs a layer that subtracts the mean from the
input. We build it by inheriting from the Block class and implementing the
`forward` method.

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

To see how it works let us feed some data into the layer.

```{.python .input  n=2}
layer = CenteredLayer()
layer(np.array([1, 2, 3, 4, 5]))
```

We can also use it to construct more complex models.

```{.python .input  n=3}
net = nn.Sequential()
net.add(nn.Dense(128), CenteredLayer())
net.initialize()
```

Let us see whether the centering layer did its job. For that we send random data through the network and check whether the mean vanishes. Note that since we are dealing with floating point numbers, we are going to see a very small albeit typically nonzero number.

```{.python .input  n=4}
y = net(np.random.uniform(size=(4, 8)))
y.mean()
```

## Layers with Parameters

Now that we know how to define layers in principle, let us define layers with parameters. These can be adjusted through training. In order to simplify things for an avid deep learning researcher the `Parameter` class and the `ParameterDict` dictionary provide some basic housekeeping functionality. In particular, they govern access, initialization, sharing, saving and loading model parameters. For instance, this way we do not need to write custom serialization routines for each new custom layer.

For instance, we can use the member variable `params` of the `ParameterDict` type that comes with the Block class. It is a dictionary that maps string type parameter names to model parameters in the `Parameter` type.  We can create a `Parameter` instance from `ParameterDict` via the `get` function.

```{.python .input  n=7}
params = gluon.ParameterDict()
params.get('param2', shape=(2, 3))
params
```

Let us use this to implement our own version of the dense layer. It has two parameters - bias and weight. To make it a bit nonstandard, we bake in the ReLU activation as default. Next, we implement a fully connected layer with both weight and bias parameters.  It uses ReLU as an activation function, where `in_units` and `units` are the number of inputs and the number of outputs, respectively.

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

Naming the parameters allows us to access them by name through dictionary lookup later. It is a good idea to give them instructive names. Next, we instantiate the `MyDense` class and access its model parameters.

```{.python .input}
dense = MyDense(units=3, in_units=5)
dense.params
```

We can directly carry out forward calculations using custom layers.

```{.python .input  n=20}
dense.initialize()
dense(np.random.uniform(size=(2, 5)))
```

We can also construct models using custom layers. Once we have that we can use it just like the built-in dense layer. The only exception is that in our case size inference is not automagic. Please consult the [MXNet documentation](http://www.mxnet.io) for details on how to do this.

```{.python .input  n=19}
net = nn.Sequential()
net.add(MyDense(8, in_units=64),
        MyDense(1, in_units=8))
net.initialize()
net(np.random.uniform(size=(2, 64)))
```

## Summary

* We can design custom layers via the Block class. This is more powerful than defining a block factory, since it can be invoked in many contexts.
* Blocks can have local parameters.


## Exercises

1. Design a layer that learns an affine transform of the data, i.e., it removes the mean and learns an additive parameter instead.
1. Design a layer that takes an input and computes a tensor reduction, i.e., it returns $y_k = \sum_{i,j} W_{ijk} x_i x_j$.
1. Design a layer that returns the leading half of the Fourier coefficients of the data. Hint - look up the `fft` function in MXNet.

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2328)

![](../img/qr_custom-layer.svg)
