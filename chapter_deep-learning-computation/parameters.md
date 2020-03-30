# Parameter Management

Once we have chosen an architecture
and set our hyperparameters,
we proceed to the training loop,
where our goal is to find parameter values
that minimize our objective function. 
After training, we will need these parameters 
in order to make future predictions.
Additionally, we will sometimes wish 
to extract the parameters 
either to reuse them in some other context,
to save our model to disk so that 
it may be exectuted in other software,
or for examination in the hopes of 
gaining scientific understanding.

Most of the time, we will be able 
to ignore the nitty-gritty details
of how parameters are declared
and manipulated, relying on Gluon
to do the heavy lifting.
However, when we move away from 
stacked architectures with standard layers, 
we will sometimes need to get into the weeds
of declaring and manipulate parameters. 
In this section, we cover the following:

* Accessing parameters for debugging, diagnostics, and visualiziations.
* Parameter initialization.
* Sharing parameters across different model components.

We start by focusing on an MLP with one hidden layer.

```{.python .input  n=1}
from mxnet import init, np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()  # Use the default initialization method

x = np.random.uniform(size=(2, 20))
net(x)  # Forward computation
```

## Parameter Access

Let us start with how to access parameters
from the models that you already know.
When a model is defined via the Sequential class,
we can first access any layer by indexing 
into the model as though it were a list.
Each layer's parameters are conveniently 
located in its `params` attribute. 
We can inspect the parameters of the `net` defined above.

```{.python .input  n=2}
print(net[0].params)
print(net[1].params)
```

The output tells us a few important things.
First, each fully-connected layer 
contains two parameters, e.g., 
`dense0_weight` and `dense0_bias`,
corresponding to that layer's 
weights and biases, respectively.
Both are stored as single precision floats.
Note that the names of the parameters
are allow us to *uniquely* identify
each layer's parameters,
even in a network contains hundreds of layers.


### Targeted Parameters

Note that each parameters is represented
as an instance of the `Parameter` class.
To do anything useful with the parameters,
we first need to access the underlying numerical values. 
There are several ways to do this.
Some are simpler while others are more general.
To begin, given a layer, 
we can access one of its parameters 
via the `bias` or `weight` attributes,
and further access that parameter's value
via its `data()` method.
The following code extracts the bias
from the second neural network layer.

```{.python .input  n=3}
print(net[1].bias)
print(net[1].bias.data())
```

Parameters are complex objects,
containing data, gradients,
and additional information.
That's why we need to request the data explicitly.
Note that the bias vector consists of zeroes
because we have not updated the network
since it was initialized.
We can also access each parameter by name,
e.g., `dense0_weight` as follows. 
Under the hood this is possible because
each layer contains a parameter dictionary. 

```{.python .input  n=4}
print(net[0].params['dense0_weight'])
print(net[0].params['dense0_weight'].data())
```

Note that unlike the biases, the weights are nonzero. 
This is because unlike biases, 
weights are initialized randomly. 
In addition to `data`, each `Parameter`
also provides a `grad()` method for 
accessing the gradient. 
It has the same shape as the weight. 
Because we have not invoked backpropagation 
for this network yet, its values are all 0.

```{.python .input  n=5}
net[0].weight.grad()
```

### All Parameters at Once

When we need to perform operations on all parameters,
accessing them one-by-one can grow tedious.
The situation can grow especially unwieldy
when we work with more complex Blocks, (e.g., nested Blocks),
since we would need to recurse 
through the entire tree in to extact
each sub-Block's parameters.
To avoid this, each Block comes 
with a `collect_params`  method 
that returns all Parameters in a single dictionary.
We can invoke `collect_params` on a single layer 
or a whole network as follows:

```{.python .input  n=6}
# parameters only for the first layer
print(net[0].collect_params())
# parameters of the entire network
print(net.collect_params())
```

This provides us with a third way of accessing the parameters of the network:

```{.python .input  n=7}
net.collect_params()['dense1_bias'].data()
```

Throughout the book we encounter Blocks 
that name their sub-Blocks in various ways. 
Sequential simply numbers them.
We can exploit this naming convention by leveraging
one clever feature of `collect_params`:
it allows us to filter the parameters 
returned by using regular expressions.

```{.python .input  n=8}
print(net.collect_params('.*weight'))
print(net.collect_params('dense0.*'))
```

### Collecting Parameters from Nested Blocks

Let us see how the parameter naming conventions work 
if we nest multiple blocks inside each other. 
For that we first define a function that produces Blocks 
(a Block factory, so to speak) and then 
combine these inside yet larger Blocks.

```{.python .input  n=20}
def block1():
    net = nn.Sequential()
    net.add(nn.Dense(32, activation='relu'))
    net.add(nn.Dense(16, activation='relu'))
    return net

def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add(block1())
    return net

rgnet = nn.Sequential()
rgnet.add(block2())
rgnet.add(nn.Dense(10))
rgnet.initialize()
rgnet(x)
```

Now that we have designed the network, 
let us see how it is organized.
Notice below that while `collect_params()`
produces a list of named parameters,
invoking `collect_params` as an attribute
reveals our network's structure.

```{.python .input}
print(rgnet.collect_params)
print(rgnet.collect_params())
```

Since the layers are hierarchically nested,
we can also access them as though 
indexing through nested lists. 
For instance, we can access the first major block, 
within it the second subblock, 
and within that the bias of the first layer,
with as follows:

```{.python .input}
rgnet[0][1][0].bias.data()
```

## Parameter Initialization

Now that we know how to access the parameters,
let us look at how to initialize them properly.
We discussed the need for initialization in :numref:`sec_numerical_stability`. 
By default, MXNet initializes weight matrices
uniformly by drawing from $U[-0.07, 0.07]$ 
and the bias parameters are all set to $0$.
However, we will often want to initialize our weights
according to various other protocols. 
MXNet's `init` module provides a variety 
of preset initialization methods.
If we want to create a custom initializer,
we need to do some extra work.

### Built-in Initialization

Let us begin by calling on built-in initializers. 
The code below initializes all parameters 
as Gaussian random variables 
with standard deviation $.01$.

```{.python .input  n=9}
# force_reinit ensures that variables are freshly initialized
# even if they were already initialized previously
net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
net[0].weight.data()[0]
```

We can also initialize all parameters 
to a given constant value (say, $1$), 
by using the initializer `Constant(1)`.

```{.python .input  n=10}
net.initialize(init=init.Constant(1), force_reinit=True)
net[0].weight.data()[0]
```

We can also apply different initialziers for certain Blocks.
For example, below we initialize the first layer
with the `Xavier` initializer
and initialize the second layer 
to a constant value of 42.

```{.python .input  n=11}
net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
net[1].initialize(init=init.Constant(42), force_reinit=True)
print(net[0].weight.data()[0])
print(net[1].weight.data()[0, 0])
```

### Custom Initialization

Sometimes, the initialization methods we need 
are not provided in the `init` module. 
In these cases, we can define a subclass of `Initializer`. 
Usually, we only need to implement the `_init_weight` function
which takes an `ndarray` argument (`data`) 
and assigns to it the desired initialized values. 
In the example below, we define an initializer
for the following strange distribution:

$$
\begin{aligned}
    w \sim \begin{cases}
        U[5, 10] & \text{ with probability } \frac{1}{4} \\
            0    & \text{ with probability } \frac{1}{2} \\
        U[-10, -5] & \text{ with probability } \frac{1}{4}
    \end{cases}
\end{aligned}
$$

```{.python .input  n=12}
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = np.random.uniform(-10, 10, data.shape)
        data *= np.abs(data) >= 5

net.initialize(MyInit(), force_reinit=True)
net[0].weight.data()[0]
```

Note that we always have the option 
of setting parameters directly by calling `data()` 
to access the underlying `ndarray`. 
A note for advanced users: 
if you want to adjust parameters within an `autograd` scope,
you need to use `set_data` to avoid confusing 
the automatic differentiation mechanics.

```{.python .input  n=13}
net[0].weight.data()[:] += 1
net[0].weight.data()[0, 0] = 42
net[0].weight.data()[0]
```

## Tied Parameters

Often, we want to share parameters across multiple layers.
Later we will see that when learning word embeddings,
it might be sensible to use the same parameters
both for encoding and decoding words. 
We discussed one such case when we introduced :numref:`sec_model_construction`. 
Let us see how to do this a bit more elegantly. 
In the following we allocate a dense layer 
and then use its parameters specifically 
to set those of another layer.

```{.python .input  n=14}
net = nn.Sequential()
# We need to give the shared layer a name such that we can reference its
# parameters
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()

x = np.random.uniform(size=(2, 20))
net(x)

# Check whether the parameters are the same
print(net[1].weight.data()[0] == net[2].weight.data()[0])
net[1].weight.data()[0, 0] = 100
# Make sure that they are actually the same object rather than just having the
# same value
print(net[1].weight.data()[0] == net[2].weight.data()[0])
```

This example shows that the parameters 
of the second and third layer are tied. 
They are not just equal, they are 
represented by the same exact `ndarray`. 
Thus, if we change one of the parameters,
the other one changes, too. 
You might wonder, 
*when parameters are tied
what happens to the gradients?*
Since the model parameters contain gradients,
the gradients of the second hidden layer
and the third hidden layer are added together
in `shared.params.grad( )` during backpropagation.

## Summary

* We have several ways to access, initialize, and tie model parameters.
* We can use custom initialization.
* Gluon has a sophisticated mechanism for accessing parameters in a unique and hierarchical manner.


## Exercises

1. Use the FancyMLP defined in :numref:`sec_model_construction` and access the parameters of the various layers.
1. Look at the [MXNet documentation](http://beta.mxnet.io/api/gluon-related/mxnet.initializer.html) and explore different initializers.
1. Try accessing the model parameters after `net.initialize()` and before `net(x)` to observe the shape of the model parameters. What changes? Why?
1. Construct a multilayer perceptron containing a shared parameter layer and train it. During the training process, observe the model parameters and gradients of each layer.
1. Why is sharing parameters a good idea?

## [Discussions](https://discuss.mxnet.io/t/2326)

![](../img/qr_parameters.svg)
