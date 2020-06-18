# Layers and Blocks
:label:`sec_model_construction`

When we first introduced neural networks,
we focused on linear models with a single output.
Here, the entire model consists of just a single neuron.
Note that a single neuron
(i) takes some set of inputs;
(ii) generates a corresponding (*scalar*) output;
and (iii) has a set of associated parameters that can be updated
to optimize some objective function of interest.
Then, once we started thinking about networks with multiple outputs,
we leveraged vectorized arithmetic
to characterize an entire *layer* of neurons.
Just like individual neurons,
layers (i) take a set of inputs,
(ii) generate corresponding outputs,
and (iii) are described by a set of tunable parameters.
When we worked through softmax regression,
a single *layer* was itself *the model*.
However, even when we subsequently
introduced multilayer perceptrons,
we could still think of the model as
retaining this same basic structure.

Interestingly, for multilayer perceptrons,
both the *entire model* and its *constituent layers*
share this structure.
The (entire) model takes in raw inputs (the features),
generates outputs (the predictions),
and possesses parameters
(the combined parameters from all constituent layers).
Likewise, each individual layer ingests inputs
(supplied by the previous layer)
generates outputs (the inputs to the subsequent layer),
and possesses a set of tunable parameters that are updated
according to the signal that flows backwards
from the subsequent layer.


While you might think that neurons, layers, and models
give us enough abstractions to go about our business,
it turns out that we often find it convenient
to speak about components that are
larger than an individual layer
but smaller than the entire model.
For example, the ResNet-152 architecture,
which is wildly popular in computer vision,
possesses hundreds of layers.
These layers consist of repeating patterns of *groups of layers*. Implementing such a network one layer at a time can grow tedious.
This concern is not just hypothetical---such
design patterns are common in practice.
The ResNet architecture mentioned above
won the 2015 ImageNet and COCO computer vision competitions
for both recognition and detection :cite:`He.Zhang.Ren.ea.2016`
and remains a go-to architecture for many vision tasks.
Similar architectures in which layers are arranged
in various repeating patterns
are now ubiquitous in other domains,
including natural language processing and speech.

To implement these complex networks,
we introduce the concept of a neural network *block*.
A block could describe a single layer,
a component consisting of multiple layers,
or the entire model itself!
One benefit of working with the block abstraction
is that they can be combined into larger artifacts,
often recursively, (see illustration in :numref:`fig_blocks`).

![Multiple layers are combined into blocks](../img/blocks.svg)
:label:`fig_blocks`

By defining code to generate blocks
of arbitrary complexity on demand,
we can write surprisingly compact code
and still implement complex neural networks.

From a software standpoint, a block is represented by a *class*.
Any subclass of it must define a forward method
that transforms its input into output
and must store any necessary parameters.
Note that some blocks do not require any parameters at all!
Finally a block must possess a backward method,
for purposes of calculating gradients.
Fortunately, due to some behind-the-scenes magic
supplied by the auto differentiation
(introduced in :numref:`sec_autograd`)
when defining our own block,
we only need to worry about parameters
and the forward function.

To begin, we revisit the codes
that we used to implement multilayer perceptrons
(:numref:`sec_mlp_gluon`).
The following code generates a network
with one fully-connected hidden layer
with 256 units and ReLU activation,
followed by a fully-connected *output layer*
with 10 units (no activation function).

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

x = np.random.uniform(size=(2, 20))

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
net(x)
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn
from torch.nn import functional as F


x = torch.randn(2,20)
net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(x)
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])

x = tf.random.uniform((2, 20))
net(x)
```

:begin_tab:`mxnet`
In this example, we constructed
our model by instantiating an `nn.Sequential`,
assigning the returned object to the `net` variable.
Next, we repeatedly call its `add` method,
appending layers in the order
that they should be executed.
In short, `nn.Sequential` defines a special kind of `Block`,
the class that presents a block in Gluon.
It maintains an ordered list of constituent `Block`s.
The `add` method simply facilitates
the addition of each successive `Block` to the list.
Note that each layer is an instance of the `Dense` class
which is itself a subclass of `Block`.
The `forward` function is also remarkably simple:
it chains each Block in the list together,
passing the output of each as the input to the next.
Note that until now, we have been invoking our models
via the construction `net(X)` to obtain their outputs.
This is actually just shorthand for `net.forward(X)`,
a slick Python trick achieved via
the Block class's `__call__` function.
:end_tab:

:begin_tab:`pytorch`
In this example, we constructed
our model by instantiating an `nn.Sequential`, with layers in the order
that they should be executed passed as arguments.
In short, `nn.Sequential` defines a special kind of `Module`,
the class that presents a block in PyTorch.
that maintains an ordered list of constituent `Module`s.
Note that each of the two fully-connected layers is an instance of the `Linear` class
which is itself a subclass of `Module`.
The `forward` function is also remarkably simple:
it chains each block in the list together,
passing the output of each as the input to the next.
Note that until now, we have been invoking our models
via the construction `net(X)` to obtain their outputs.
This is actually just shorthand for `net.forward(X)`,
a slick Python trick achieved via
the Block class's `__call__` function.
:end_tab:

:begin_tab:`tensorflow`
In this example, we constructed
our model by instantiating an `keras.models.Sequential`, with layers in the order
that they should be executed passed as arguments.
In short, `Sequential` defines a special kind of `keras.Model`,
the class that presents a block in Keras.
It maintains an ordered list of constituent `Model`s.
Note that each of the two fully-connected layers is an instance of the `Dense` class
which is itself a subclass of `Model`.
The forward function is also remarkably simple:
it chains each block in the list together,
passing the output of each as the input to the next.
Note that until now, we have been invoking our models
via the construction `net(X)` to obtain their outputs.
This is actually just shorthand for `net.call(X)`,
a slick Python trick achieved via
the Block class's `__call__` function.
:end_tab:

## A Custom Block

Perhaps the easiest way to develop intuition
about how a block works
is to implement one ourselves.
Before we implement our own custom block,
we briefly summarize the basic functionality
that each block must provide:

1. Ingest input data as arguments to its forward method.
1. Generate an output by having forward return a value.
   Note that the output may have a different shape from the input.      For example, the first fully-connected layer in our model above ingests an      input of arbitrary dimension but returns
   an output of dimension 256.
1. Calculate the gradient of its output with respect to its input,      which can be accessed via its backward method.
   Typically this happens automatically.
1. Store and provide access to those parameters necessary
   to execute the forward computation.
1. Initialize these parameters as needed.

In the following snippet,
we code up a block from scratch
corresponding to a multilayer perceptron
with one hidden layer with 256 hidden nodes,
and a 10-dimensional output layer.
Note that the `MLP` class below inherits the class represents a block.
We will rely heavily on the parent class's methods,
supplying only our own `__init__` and forward methods.

```{.python .input}
class MLP(nn.Block):
    # Declare a layer with model parameters. Here, we declare two fully
    # connected layers
    def __init__(self, **kwargs):
        # Call the constructor of the MLP parent class Block to perform the
        # necessary initialization. In this way, other function parameters can
        # also be specified when constructing an instance, such as the model
        # parameter, params, described in the following sections
        super().__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')  # Hidden layer
        self.out = nn.Dense(10)  # Output layer

    # Define the forward computation of the model, that is, how to return the
    # required model output based on the input x
    def forward(self, x):
        return self.out(self.hidden(x))
```

```{.python .input}
#@tab pytorch
class MLP(nn.Module):
    # Declare a layer with model parameters. Here, we declare two fully
    # connected layers
    def __init__(self):
        # Call the constructor of the MLP parent class Block to perform the
        # necessary initialization. In this way, other function parameters can
        # also be specified when constructing an instance, such as the model
        # parameter, params, described in the following sections
        super().__init__()
        self.hidden = nn.Linear(20,256)  # Hidden layer
        self.out = nn.Linear(256,10)  # Output layer

    # Define the forward computation of the model, that is, how to return the
    # required model output based on the input x
    def forward(self, x):
        # Note here we use the funtional version of ReLU defined in the
        # nn.functional module.
        return self.out(F.relu(self.hidden(x)))
```

```{.python .input}
#@tab tensorflow
class MLP(tf.keras.Model):
    # Declare a layer with model parameters. Here, we declare two fully
    # connected layers
    def __init__(self):
        # Call the constructor of the MLP parent class Block to perform the
        # necessary initialization. In this way, other function parameters can
        # also be specified when constructing an instance, such as the model
        # parameter, params, described in the following sections
        super().__init__()
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)  # Hidden layer
        self.out = tf.keras.layers.Dense(units=10)  # Output layer

    # Define the forward computation of the model, that is, how to return the
    # required model output based on the input x
    def call(self, x):
        return self.out(self.hidden((x)))
```

To begin, let us focus on the forward method.
Note that it takes `x` as input,
calculates the hidden representation (`self.hidden(x)`) with the activation function applied,
and outputs its logits (`self.out( ... )`).
In this MLP implementation,
both layers are instance variables.
To see why this is reasonable, imagine
instantiating two MLPs, `net1` and `net2`,
and training them on different data.
Naturally, we would expect them
to represent two different learned models.

We instantiate the MLP's layers
in the `__init__` method (the constructor)
and subsequently invoke these layers
on each call to the forward method.
Note a few key details.
First, our customized `__init__` method
invokes the parent class's `__init__` method
via `super().__init__()`
sparing us the pain of restating
boilerplate code applicable to most Blocks.
We then instantiate our two fully-connected layers,
assigning them to `self.hidden` and `self.out`.
Note that unless we implement a new operator,
we need not worry about backpropagation (the backward method)
or parameter initialization.
The system will generate these methods automatically.
Let us try this out:

```{.python .input}
net = MLP()
net.initialize()
net(x)
```

```{.python .input}
#@tab pytorch
net = MLP()
net(x)
```

```{.python .input}
#@tab tensorflow
net = MLP()
net(x)
```

A key virtue of the block abstraction is its versatility.
We can subclass the block class to create layers
(such as the fully-connected layer class),
entire models (such as the `MLP` above),
or various components of intermediate complexity.
We exploit this versatility
throughout the following chapters,
especially when addressing
convolutional neural networks.


## The Sequential Block

We can now take a closer look
at how the `Sequential` class works.
Recall that `Sequential` was designed
to daisy-chain other blocks together.
To build our own simplified `MySequential`,
we just need to define two key methods:
1. A method to append blocks one by one to a list.
2. A forward method to pass an input through the chain of Blocks
(in the same order as they were appended).

The following `MySequential` class delivers the same
functionality the default `Sequential` class:

```{.python .input}
class MySequential(nn.Block):
    def add(self, block):
        # Here, block is an instance of a Block subclass, and we assume it has
        # a unique name. We save it in the member variable _children of the
        # Block class, and its type is OrderedDict. When the MySequential
        # instance calls the initialize function, the system automatically
        # initializes all members of _children
        self._children[block.name] = block

    def forward(self, x):
        # OrderedDict guarantees that members will be traversed in the order
        # they were added
        for block in self._children.values():
            x = block(x)
        return x
```

```{.python .input}
#@tab pytorch
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for block in args:
            # Here, block is an instance of a Module subclass. We save it in the
            # member variable _modules of the Module class, and its type is
            # OrderedDict.
            self._modules[block] = block

    def forward(self, x):
        # OrderedDict guarantees that members will be traversed in the order
        # they were added
        for block in self._modules.values():
            x = block(x)
        return x
```

```{.python .input}
#@tab tensorflow
class MySequential(tf.keras.Model):
    def __init__(self, *args):
        super().__init__()
        self._modules = tf.keras.Sequential()
        for block in args:
            # Here, block is an instance of a Model subclass.
            self._modules.add(block)

    def call(self, x):
        return self._modules(x)
```

:begin_tab:`mxnet`
The `add` method adds a single block
to the ordered dictionary `_children`.
You might wonder why every Gluon `Block`
possesses a `_children` attribute
and why we used it rather than just
defining a Python list ourselves.
In short the chief advantage of `_children`
is that during our block's parameter initialization,
Gluon knows to look in the `_children`
dictionary to find sub-Blocks whose
parameters also need to be initialized.
:end_tab:

:begin_tab:`pytorch`
In the `__init__` method, we add every block
to the ordered dictionary `_modules` one by one.
You might wonder why every `Module`
possesses a `_modules` attribute
and why we used it rather than just
defining a Python list ourselves.
In short the chief advantage of `_modules`
is that during our block's parameter initialization,
the system knows to look in the `_modules`
dictionary to find sub-blocks whose
parameters also need to be initialized.
:end_tab:

:begin_tab:`tensorflow`
FIXME, don't use `Sequential` to implement `MySequential`.
:end_tab:

When our `MySequential`'s forward method is invoked,
each added block is executed
in the order in which they were added.
We can now reimplement an MLP
using our `MySequential` class.

```{.python .input}
net = MySequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
net(x)
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(x)
```

```{.python .input}
#@tab tensorflow
net = MySequential(
    tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.relu))
net(x)
```

Note that this use of `MySequential`
is identical to the code we previously wrote
for the `Sequential` class
(as described in :numref:`sec_mlp_gluon`).


## Executing Code in the forward Method

The `Sequential` class makes model construction easy,
allowing us to assemble new architectures
without having to define our own class.
However, not all architectures are simple daisy chains.
When greater flexibility is required,
we will want to define our own blocks.
For example, we might want to execute
Python's control flow within the forward method.
Moreover we might want to perform
arbitrary mathematical operations,
not simply relying on predefined neural network layers.

You might have noticed that until now,
all of the operations in our networks
have acted upon our network's activations
and its parameters.
Sometimes, however, we might want to
incorporate terms
that are neither the result of previous layers
nor updatable parameters.
We call these *constant* parameters.
Say for example that we want a layer
that calculates the function
$f(\mathbf{x},\mathbf{w}) = c \cdot \mathbf{w}^\top \mathbf{x}$,
where $\mathbf{x}$ is the input, $\mathbf{w}$ is our parameter,
and $c$ is some specified constant
that is not updated during optimization.

```{.python .input}
class FixedHiddenMLP(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Random weight parameters created with the get_constant are not
        # iterated during training (i.e., constant parameters)
        self.rand_weight = self.params.get_constant(
            'rand_weight', np.random.uniform(size=(20, 20)))
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, x):
        x = self.dense(x)
        # Use the constant parameters created, as well as the relu
        # and dot functions
        x = npx.relu(np.dot(x, self.rand_weight.data()) + 1)
        # Reuse the fully connected layer. This is equivalent to sharing
        # parameters with two fully connected layers
        x = self.dense(x)
        # Here in Control flow, we need to call asscalar to return the scalar
        # for comparison
        while np.abs(x).sum() > 1:
            x /= 2
        return x.sum()
```

```{.python .input}
#@tab pytorch
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Random weight parameters that will not compute gradients and
        # and therefore keep constant during training.
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, x):
        x = self.linear(x)
        # Use the constant parameters created, as well as the relu
        # and dot functions
        x = F.relu(torch.mm(x, self.rand_weight) + 1)
        # Reuse the fully connected layer. This is equivalent to sharing
        # parameters with two fully connected layers
        x = self.linear(x)
        # Here in Control flow, we need to call asscalar to return the scalar
        # for comparison
        while x.norm().item() > 1:
            x /= 2
        return x.sum()
```

```{.python .input}
#@tab tensorflow
class FixedHiddenMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        # Random weight parameters that will not compute gradients and
        # and therefore keep constant during training.
        self.rand_weight = tf.constant(tf.random.uniform((20, 20)))
        self.dense = tf.keras.layers.Dense(20, activation=tf.nn.relu)

    def call(self, inputs):
        x = self.flatten(inputs)
        # Use the constant parameters created, as well as the relu
        # and dot functions
        x = tf.nn.relu(tf.matmul(x, self.rand_weight) + 1)
        # Reuse the fully connected layer. This is equivalent to sharing
        # parameters with two fully connected layers
        x = self.dense(x)
        # Here in Control flow, we need to call asscalar to return the scalar
        # for comparison
        while tf.norm(x) > 1:
            x /= 2
        return tf.reduce_sum(x)
```

In this `FixedHiddenMLP` model,
we implement a hidden layer whose weights
(`self.rand_weight`) are initialized randomly
at instantiation and are thereafter constant.
This weight is not a model parameter
and thus it is never updated by backpropagation.
The network then passes the output of this *fixed* layer
through a fully-connected layer.

Note that before returning output,
our model did something unusual.
We ran a `while` loop, testing
on the condition it's norm is larger than 1,
and dividing our output vector by $2$
until it satisfied the condition.
Finally, we returned the sum of the entries in `x`.
To our knowledge, no standard neural network
performs this operation.
Note that this particular operation may not be useful
in any real world task.
Our point is only to show you how to integrate
arbitrary code into the flow of your
neural network computations.

```{.python .input}
net = FixedHiddenMLP()
net.initialize()
net(x)
```

```{.python .input}
#@tab pytorch, tensorflow
net = FixedHiddenMLP()
net(x)
```

We can mix and match various
ways of assembling blocks together.
In the following example, we nest blocks
in some creative ways.

```{.python .input}
class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation='relu'),
                     nn.Dense(32, activation='relu'))
        self.dense = nn.Dense(16, activation='relu')

    def forward(self, x):
        return self.dense(self.net(x))

chimera = nn.Sequential()
chimera.add(NestMLP(), nn.Dense(20), FixedHiddenMLP())

chimera.initialize()
chimera(x)
```

```{.python .input}
#@tab pytorch
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, x):
        return self.linear(self.net(x))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
chimera(x)
```

```{.python .input}
#@tab tensorflow
class NestMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.net = tf.keras.Sequential()
        self.net.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        self.net.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
        self.dense = tf.keras.layers.Dense(16, activation=tf.nn.relu)

    def call(self, inputs):
        return self.dense(self.net(inputs))

chimera = tf.keras.Sequential()
chimera.add(NestMLP())
chimera.add(tf.keras.layers.Dense(20))
chimera.add(FixedHiddenMLP())
chimera(x)
```

## Compilation

:begin_tab:`mxnet, tensorflow`
The avid reader might start to worry
about the efficiency of some of these operations.
After all, we have lots of dictionary lookups,
code execution, and lots of other Pythonic things
taking place in what is supposed to be
a high performance deep learning library.
The problems of Python's [Global Interpreter Lock](https://wiki.python.org/moin/GlobalInterpreterLock) are well known. In the context of deep learning,
we worry that our extremely fast GPU(s)
might have to wait until a puny CPU
runs Python code before it gets another job to run.
The best way to speed up Python is by avoiding it altogether.
One way that Gluon does this by allowing for
Hybridization (:numref:`sec_hybridize`).
Here, the Python interpreter executes a Block
the first time it is invoked.
The Gluon runtime records what is happening
and the next time around it short-circuits calls to Python.
This can accelerate things considerably in some cases
but care needs to be taken when control flow (as above)
leads down different branches on different passes through the net.
We recommend that the interested reader check out
the hybridization section (:numref:`sec_hybridize`)
to learn about compilation after finishing the current chapter.
:end_tab:

## Summary

* Layers are blocks.
* Many layers can comprise a block.
* Many blocks can comprise a block.
* A block can contain code.
* Blocks take care of lots of housekeeping, including parameter initialization and backpropagation.
* Sequential concatenations of layers and blocks are handled by the `Sequential` Block.


## Exercises

1. What kinds of problems will occur if you change `MySequential` to store blocks in a Python list.
1. Implement a block that takes two blocks as an argument, say `net1` and `net2` and returns the concatenated output of both networks in the forward pass (this is also called a parallel block).
1. Assume that you want to concatenate multiple instances of the same network. Implement a factory function that generates multiple instances of the same block and build a larger network from it.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/54)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/55)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/264)
:end_tab:
