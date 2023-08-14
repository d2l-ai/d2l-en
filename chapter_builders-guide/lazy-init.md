```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Lazy Initialization
:label:`sec_lazy_init`

So far, it might seem that we got away
with being sloppy in setting up our networks.
Specifically, we did the following unintuitive things,
which might not seem like they should work:

* We defined the network architectures
  without specifying the input dimensionality.
* We added layers without specifying
  the output dimension of the previous layer.
* We even "initialized" these parameters
  before providing enough information to determine
  how many parameters our models should contain.

You might be surprised that our code runs at all.
After all, there is no way the deep learning framework
could tell what the input dimensionality of a network would be.
The trick here is that the framework *defers initialization*,
waiting until the first time we pass data through the model,
to infer the sizes of each layer on the fly.


Later on, when working with convolutional neural networks,
this technique will become even more convenient
since the input dimensionality
(e.g., the resolution of an image)
will affect the dimensionality
of each subsequent layer.
Hence the ability to set parameters
without the need to know,
at the time of writing the code,
the value of the dimension
can greatly simplify the task of specifying
and subsequently modifying our models.
Next, we go deeper into the mechanics of initialization.

```{.python .input}
%%tab mxnet
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

To begin, let's instantiate an MLP.

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(256), nn.relu, nn.Dense(10)])
```

At this point, the network cannot possibly know
the dimensions of the input layer's weights
because the input dimension remains unknown.

:begin_tab:`mxnet, pytorch, tensorflow`
Consequently the framework has not yet initialized any parameters.
We confirm by attempting to access the parameters below.
:end_tab:

:begin_tab:`jax`
As mentioned in :numref:`subsec_param-access`, parameters and the network definition are decoupled
in Jax and Flax, and the user handles both manually. Flax models are stateless
hence there is no `parameters` attribute.
:end_tab:

```{.python .input}
%%tab mxnet
print(net.collect_params)
print(net.collect_params())
```

```{.python .input}
%%tab pytorch
net[0].weight
```

```{.python .input}
%%tab tensorflow
[net.layers[i].get_weights() for i in range(len(net.layers))]
```

:begin_tab:`mxnet`
Note that while the parameter objects exist,
the input dimension to each layer is listed as -1.
MXNet uses the special value -1 to indicate
that the parameter dimension remains unknown.
At this point, attempts to access `net[0].weight.data()`
would trigger a runtime error stating that the network
must be initialized before the parameters can be accessed.
Now let's see what happens when we attempt to initialize
parameters via the `initialize` method.
:end_tab:

:begin_tab:`tensorflow`
Note that each layer objects exist but the weights are empty.
Using `net.get_weights()` would throw an error since the weights
have not been initialized yet.
:end_tab:

```{.python .input}
%%tab mxnet
net.initialize()
net.collect_params()
```

:begin_tab:`mxnet`
As we can see, nothing has changed.
When input dimensions are unknown,
calls to initialize do not truly initialize the parameters.
Instead, this call registers to MXNet that we wish
(and optionally, according to which distribution)
to initialize the parameters.
:end_tab:

Next let's pass data through the network
to make the framework finally initialize parameters.

```{.python .input}
%%tab mxnet
X = np.random.uniform(size=(2, 20))
net(X)

net.collect_params()
```

```{.python .input}
%%tab pytorch
X = torch.rand(2, 20)
net(X)

net[0].weight.shape
```

```{.python .input}
%%tab tensorflow
X = tf.random.uniform((2, 20))
net(X)
[w.shape for w in net.get_weights()]
```

```{.python .input}
%%tab jax
params = net.init(d2l.get_key(), jnp.zeros((2, 20)))
jax.tree_util.tree_map(lambda x: x.shape, params).tree_flatten_with_keys()
```

As soon as we know the input dimensionality,
20,
the framework can identify the shape of the first layer's weight matrix by plugging in the value of 20.
Having recognized the first layer's shape, the framework proceeds
to the second layer,
and so on through the computational graph
until all shapes are known.
Note that in this case,
only the first layer requires lazy initialization,
but the framework initializes sequentially.
Once all parameter shapes are known,
the framework can finally initialize the parameters.

:begin_tab:`pytorch`
The following method
passes in dummy inputs
through the network
for a dry run
to infer all parameter shapes
and subsequently initializes the parameters.
It will be used later when default random initializations are not desired.
:end_tab:

:begin_tab:`jax`
Parameter initialization in Flax is always done manually and handled by the
user. The following method takes a dummy input and a key dictionary as argument.
This key dictionary has the rngs for initializing the model parameters
and dropout rng for generating the dropout mask for the models with
dropout layers. More about dropout will be covered later in :numref:`sec_dropout`.
Ultimately the method initializes the model returning the parameters.
We have been using it under the hood in the previous sections as well.
:end_tab:

```{.python .input}
%%tab pytorch
@d2l.add_to_class(d2l.Module)  #@save
def apply_init(self, inputs, init=None):
    self.forward(*inputs)
    if init is not None:
        self.net.apply(init)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Module)  #@save
def apply_init(self, dummy_input, key):
    params = self.init(key, *dummy_input)  # dummy_input tuple unpacked
    return params
```

## Summary

Lazy initialization can be convenient, allowing the framework to infer parameter shapes automatically, making it easy to modify architectures and eliminating one common source of errors.
We can pass data through the model to make the framework finally initialize parameters.


## Exercises

1. What happens if you specify the input dimensions to the first layer but not to subsequent layers? Do you get immediate initialization?
1. What happens if you specify mismatching dimensions?
1. What would you need to do if you have input of varying dimensionality? Hint: look at the parameter tying.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/280)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/8092)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/281)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17992)
:end_tab:
