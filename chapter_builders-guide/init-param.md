# Parameter Initialization

Now that we know how to access the parameters,
let's look at how to initialize them properly.
We discussed the need for proper initialization in :numref:`sec_numerical_stability`.
The deep learning framework provides default random initializations to its layers.
However, we often want to initialize our weights
according to various other protocols. The framework provides most commonly
used protocols, and also allows to create a custom initializer.

:begin_tab:`mxnet`
By default, MXNet initializes weight parameters by randomly drawing from a uniform distribution $U(-0.07, 0.07)$,
clearing bias parameters to zero.
MXNet's `init` module provides a variety
of preset initialization methods.
:end_tab:

:begin_tab:`pytorch`
By default, PyTorch initializes weight and bias matrices
uniformly by drawing from a range that is computed according to the input and output dimension.
PyTorch's `nn.init` module provides a variety
of preset initialization methods.
:end_tab:

:begin_tab:`tensorflow`
By default, Keras initializes weight matrices uniformly by drawing from a range that is computed according to the input and output dimension, and the bias parameters are all set to zero.
TensorFlow provides a variety of initialization methods both in the root module and the `keras.initializers` module.
:end_tab:

```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

```{.python .input  n=2}
%%tab mxnet
from mxnet import init, np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
net.add(nn.Dense(8, activation='relu'))
net.add(nn.Dense(1))
net.initialize()  # Use the default initialization method

X = np.random.uniform(size=(2, 4))
net(X).shape
```

```{.python .input  n=3}
%%tab pytorch
import torch
from torch import nn

net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(), nn.LazyLinear(1))
X = torch.rand(size=(2, 4))
net(X).shape
```

```{.python .input  n=4}
%%tab tensorflow
import tensorflow as tf

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation=tf.nn.relu),
    tf.keras.layers.Dense(1),
])

X = tf.random.uniform((2, 4))
net(X).shape
```

## [**Built-in Initialization**]

Let's begin by calling on built-in initializers.
The code below initializes all weight parameters
as Gaussian random variables
with standard deviation 0.01, while bias parameters cleared to zero.

```{.python .input  n=5}
%%tab mxnet
# Here `force_reinit` ensures that parameters are freshly initialized even if
# they were already initialized previously
net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
net[0].weight.data()[0]
```

```{.python .input  n=6}
%%tab pytorch
def init_normal(module):
    if type(module) == nn.Linear:
        nn.init.normal_(module.weight, mean=0, std=0.01)
        nn.init.zeros_(module.bias)
net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
```

```{.python .input  n=7}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1)])

net(X)
net.weights[0], net.weights[1]
```

We can also initialize all the parameters
to a given constant value (say, 1).

```{.python .input  n=8}
%%tab mxnet
net.initialize(init=init.Constant(1), force_reinit=True)
net[0].weight.data()[0]
```

```{.python .input  n=9}
%%tab pytorch
def init_constant(module):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, 1)
        nn.init.zeros_(module.bias)
net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]
```

```{.python .input  n=10}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.Constant(1),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1),
])

net(X)
net.weights[0], net.weights[1]
```

[**We can also apply different initializers for certain blocks.**]
For example, below we initialize the first layer
with the Xavier initializer
and initialize the second layer
to a constant value of 42.

```{.python .input  n=11}
%%tab mxnet
net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
net[1].initialize(init=init.Constant(42), force_reinit=True)
print(net[0].weight.data()[0])
print(net[1].weight.data())
```

```{.python .input  n=12}
%%tab pytorch
def init_xavier(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)
def init_42(module):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
```

```{.python .input  n=13}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.GlorotUniform()),
    tf.keras.layers.Dense(
        1, kernel_initializer=tf.keras.initializers.Constant(42)),
])

net(X)
print(net.layers[1].weights[0])
print(net.layers[2].weights[0])
```

### [**Custom Initialization**]

Sometimes, the initialization methods we need
are not provided by the deep learning framework.
In the example below, we define an initializer
for any weight parameter $w$ using the following strange distribution:

$$
\begin{aligned}
    w \sim \begin{cases}
        U(5, 10) & \text{ with probability } \frac{1}{4} \\
            0    & \text{ with probability } \frac{1}{2} \\
        U(-10, -5) & \text{ with probability } \frac{1}{4}
    \end{cases}
\end{aligned}
$$

:begin_tab:`mxnet`
Here we define a subclass of the `Initializer` class.
Usually, we only need to implement the `_init_weight` function
which takes a tensor argument (`data`)
and assigns to it the desired initialized values.
:end_tab:

:begin_tab:`pytorch`
Again, we implement a `my_init` function to apply to `net`.
:end_tab:

:begin_tab:`tensorflow`
Here we define a subclass of `Initializer` and implement the `__call__`
function that return a desired tensor given the shape and data type.
:end_tab:

```{.python .input  n=14}
%%tab mxnet
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = np.random.uniform(-10, 10, data.shape)
        data *= np.abs(data) >= 5

net.initialize(MyInit(), force_reinit=True)
net[0].weight.data()[:2]
```

```{.python .input  n=15}
%%tab pytorch
def my_init(module):
    if type(module) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in module.named_parameters()][0])
        nn.init.uniform_(module.weight, -10, 10)
        module.weight.data *= module.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]
```

```{.python .input  n=16}
%%tab tensorflow
class MyInit(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        data=tf.random.uniform(shape, -10, 10, dtype=dtype)
        factor=(tf.abs(data) >= 5)
        factor=tf.cast(factor, tf.float32)
        return data * factor

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=MyInit()),
    tf.keras.layers.Dense(1),
])

net(X)
print(net.layers[1].weights[0])
```

Note that we always have the option
of setting parameters directly.

```{.python .input  n=17}
%%tab mxnet
net[0].weight.data()[:] += 1
net[0].weight.data()[0, 0] = 42
net[0].weight.data()[0]
```

```{.python .input  n=18}
%%tab pytorch
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
```

```{.python .input  n=19}
%%tab tensorflow
net.layers[1].weights[0][:].assign(net.layers[1].weights[0] + 1)
net.layers[1].weights[0][0, 0].assign(42)
net.layers[1].weights[0]
```

## Summary

We can initialize parameters using built-in and custom initializers.

## Exercises

Look up the online documentation for more built-in initializers.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/8089)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/8090)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/8091)
:end_tab:
