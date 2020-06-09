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
it may be executed in other software,
or for examination in the hopes of 
gaining scientific understanding.

Most of the time, we will be able 
to ignore the nitty-gritty details
of how parameters are declared
and manipulated, relying on the framework
to do the heavy lifting.
However, when we move away from 
stacked architectures with standard layers, 
we will sometimes need to get into the weeds
of declaring and manipulating parameters. 
In this section, we cover the following:

* Accessing parameters for debugging, diagnostics, and visualizations.
* Parameter initialization.
* Sharing parameters across different model components.

We start by focusing on an MLP with one hidden layer.

```{.python .input}
from mxnet import init, np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
net.add(nn.Dense(8, activation='relu'))
net.add(nn.Dense(1))
net.initialize()  # Use the default initialization method

x = np.random.uniform(size=(2, 4))
net(x)  # Forward computation
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
x = torch.randn(2, 4)
net(x)
```

## Parameter Access

Let us start with how to access parameters
from the models that you already know.
When a model is defined via the Sequential class,
we can first access any layer by indexing 
into the model as though it were a list.
Each layer's parameters are conveniently 
located in its attribute. 
We can inspect the parameters of the `net` defined above as a dictionary.

```{.python .input}
print(net[0].params)
print(net[1].params)
```

```{.python .input}
#@tab pytorch
print(net[2].state_dict())  
```

The output tells us a few important things.
First, each fully-connected layer 
contains two parameters, e.g., 
`weight` and `bias` (may with prefix),
corresponding to that layer's 
weights and biases, respectively.
Both are stored as single precision floats.
Note that the names of the parameters
allow us to *uniquely* identify
each layer's parameters,
even in a network containing hundreds of layers.


### Targeted Parameters

Note that each parameter is represented
as an instance of the `Parameter` class.
To do anything useful with the parameters,
we first need to access the underlying numerical values. 
There are several ways to do this.
Some are simpler while others are more general.
To begin, given a layer, 
we can access one of its parameters 
via the `bias` or `weight` attributes, which returns an `Parameter` instance
and further access that parameter's value
via its `data` method.
The following code extracts the bias
from the second neural network layer.

```{.python .input}
print(type(net[1].bias))
print(net[1].bias)
print(net[1].bias.data())
```

```{.python .input}
#@tab pytorch
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
```

Parameters are complex objects,
containing data, gradients,
and additional information.
That's why we need to request the data explicitly.


In addition to `data`, each `Parameter` also provides a `grad` method for accessing the gradient. Because we have not invoked backpropagation for this network yet, it is in its initial state.

```{.python .input}
net[0].weight.grad()
```

```{.python .input}
#@tab pytorch
net[0].weight.grad == None
```

### All Parameters at Once

When we need to perform operations on all parameters,
accessing them one-by-one can grow tedious.
The situation can grow especially unwieldy
when we work with more complex blocks, (e.g., nested Blocks),
since we would need to recurse 


through the entire tree in to extract
each sub-block's parameters.

```{.python .input}
# parameters only for the first layer
print(net[0].collect_params())
# parameters of the entire network
print(net.collect_params())
```

```{.python .input}
#@tab pytorch
# parameters only for the first layer
print(net[0].state_dict())
# parameters of the entire network
print(net.state_dict())
```

This provides us with another way of accessing the parameters of the network:

```{.python .input}
net.collect_params()['dense1_bias'].data()
```

```{.python .input}
#@tab pytorch
net.state_dict()['2.bias'].data
```

:begin_tab:`mxnet`
Throughout the book we encounter Blocks 
that name their sub-Blocks in various ways. 
Sequential simply numbers them.
We can exploit this naming convention by leveraging
one clever feature of `collect_params`:
it allows us to filter the parameters 
returned by using regular expressions.
:end_tab:

```{.python .input}
print(net.collect_params('.*weight'))
print(net.collect_params('dense0.*'))
```

### Collecting Parameters from Nested Blocks

Let us see how the parameter naming conventions work 
if we nest multiple blocks inside each other. 
For that we first define a function that produces blocks 
(a block factory, so to speak) and then 
combine these inside yet larger blocks.

```{.python .input}
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

```{.python .input}
#@tab pytorch
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(x)
```

Now that we have designed the network, 
let us see how it is organized.

```{.python .input}
print(rgnet.collect_params)
print(rgnet.collect_params())
```

```{.python .input}
#@tab pytorch
print(rgnet)
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

```{.python .input}
#@tab pytorch
rgnet[0][1][0].bias.data
```

## Parameter Initialization

Now that we know how to access the parameters,
let us look at how to initialize them properly.
We discussed the need for initialization in :numref:`sec_numerical_stability`. 

:begin_tab:`mxnet`
By default, MXNet initializes weight matrices
uniformly by drawing from $U[-0.07, 0.07]$ 
and the bias parameters are all set to $0$.
However, we will often want to initialize our weights
according to various other protocols. 
MXNet's `init` module provides a variety 
of preset initialization methods.
If we want to create a custom initializer,
we need to do some extra work.
:end_tab:

:begin_tab:`pytorch`
By default, PyTorch initializes weight and bias matrices
uniformly by drawing from a range that is computed according to the input and output dimension. 
However, we will often want to initialize our weights
according to various other protocols. 
PyTorch's `nn.init` module provides a variety 
of preset initialization methods.
If we want to create a custom initializer,
we need to do some extra work.
:end_tab:

### Built-in Initialization

Let us begin by calling on built-in initializers. 
The code below initializes all weight parameters 
as Gaussian random variables 
with standard deviation $.01$, while bias parameters set to 0.

```{.python .input}
# force_reinit ensures that variables are freshly initialized
# even if they were already initialized previously
net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
net[0].weight.data()[0]
```

```{.python .input}
#@tab pytorch 
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(net[0].weight, mean=0, std=0.01)
        nn.init.zeros_(net[0].bias)
net.apply(init_normal)    
net[0].weight.data[0], net[0].bias.data[0]
```

We can also initialize all parameters 
to a given constant value (say, $1$).

```{.python .input}
net.initialize(init=init.Constant(1), force_reinit=True)
net[0].weight.data()[0]
```

```{.python .input}
#@tab pytorch
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.constant_(net[0].weight, 1)
        nn.init.zeros_(net[0].bias)
net.apply(init_normal)    
net[0].weight.data[0], net[0].bias.data[0]
```

We can also apply different initializers for certain Blocks.
For example, below we initialize the first layer
with the `Xavier` initializer
and initialize the second layer 
to a constant value of 42.

```{.python .input}
net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
net[1].initialize(init=init.Constant(42), force_reinit=True)
print(net[0].weight.data()[0])
print(net[1].weight.data())
```

```{.python .input}
#@tab pytorch
def xavier(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        torch.nn.init.constant_(m.weight, 42)
        
net[0].apply(xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
```

### Custom Initialization

Sometimes, the initialization methods we need 
are not provided in the `init` module. 
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

:begin_tab:`mxnet`

Here we define a subclass of `Initializer`. 
Usually, we only need to implement the `_init_weight` function
which takes a tensor argument (`data`) 
and assigns to it the desired initialized values. 

:end_tab:

:begin_tab:`pytorch`
Again, we implement a `my_init` function to apply to `net`.
:end_tab:

```{.python .input}
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = np.random.uniform(-10, 10, data.shape)
        data *= np.abs(data) >= 5

net.initialize(MyInit(), force_reinit=True)
net[0].weight.data()[0:2]
```

```{.python .input}
#@tab pytorch
def my_init(m):
    if type(m) == nn.Linear:
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5    
    
net.apply(my_init)
net[0].weight[0:2]
```

Note that we always have the option 
of setting parameters directly by calling `data` 
to access the underlying data. 

:begin_tab:`mxnet`
A note for advanced users: 
if you want to adjust parameters within an `autograd` scope,
you need to use `set_data` to avoid confusing 
the automatic differentiation mechanics.
:end_tab:

```{.python .input}
net[0].weight.data()[:] += 1
net[0].weight.data()[0, 0] = 42
net[0].weight.data()[0]
```

```{.python .input}
#@tab pytorch
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
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

```{.python .input}
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

```{.python .input}
#@tab pytorch
# We need to give the shared layer a name such that we can reference its
# parameters
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), 
                    shared, nn.ReLU(), 
                    shared, nn.ReLU(), 
                    nn.Linear(8, 1))
net(x)
# Check whether the parameters are the same
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# Make sure that they are actually the same object rather than just having the
# same value
print(net[2].weight.data[0] == net[4].weight.data[0])
```

This example shows that the parameters 
of the second and third layer are tied. 
They are not just equal, they are 
represented by the same exact tensor. 
Thus, if we change one of the parameters,
the other one changes, too. 
You might wonder, 
*when parameters are tied
what happens to the gradients?*
Since the model parameters contain gradients,
the gradients of the second hidden layer
and the third hidden layer are added together
during backpropagation.

## Summary

* We have several ways to access, initialize, and tie model parameters.
* We can use custom initialization.


## Exercises

1. Use the FancyMLP defined in :numref:`sec_model_construction` and access the parameters of the various layers.
1. Look at the `init` module document to explore different initializers.
1. Construct a multilayer perceptron containing a shared parameter layer and train it. During the training process, observe the model parameters and gradients of each layer.
1. Why is sharing parameters a good idea?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/56)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/57)
:end_tab:
