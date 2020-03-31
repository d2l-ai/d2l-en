# File I/O

So far we discussed how to process data and how 
to build, train, and test deep learning models. 
However, at some point, we will hopefully be happy enough
with the learned models that we will want 
to save the results for later use in various contexts
(perhaps even to make predictions in deployment). 
Additionally, when running a long training process,
the best practice is to periodically save intermediate results (checkpointing)
to ensure that we do not lose several days worth of computation
if we trip over the power cord of our server.
Thus it is time we learned how to load and store 
both individual weight vectors and entire models. 
This section addresses both issues.

## Loading and Saving `ndarray`s

For individual `ndarray`s, we can directly 
invoke their `load` and `save` functions 
to read and write them respectively. 
Both functions require that we supply a name,
and `save` requires as input the variable to be saved.

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

x = np.arange(4)
npx.save('x-file', x)
```

We can now read this data from the stored file back into memory.

```{.python .input}
x2 = npx.load('x-file')
x2
```

MXNet also allows us to store a list of `ndarray`s and read them back into memory.

```{.python .input  n=2}
y = np.zeros(4)
npx.save('x-files', [x, y])
x2, y2 = npx.load('x-files')
(x2, y2)
```

We can even write and read a dictionary that maps 
from strings to `ndarray`s. 
This is convenient when we want 
to read or write all the weights in a model.

```{.python .input  n=4}
mydict = {'x': x, 'y': y}
npx.save('mydict', mydict)
mydict2 = npx.load('mydict')
mydict2
```

## Gluon Model Parameters

Saving individual weight vectors (or other `ndarray` tensors) is useful 
but it gets very tedious if we want to save 
(and later load) an entire model.
After all, we might have hundreds of 
parameter groups sprinkled throughout. 
For this reason Gluon provides built-in functionality 
to load and save entire networks.
An important detail to note is that this 
saves model *parameters* and not the entire model. 
For example, if we have a 3 layer MLP,
we need to specify the *architecture* separately. 
The reason for this is that the models themselves can contain arbitrary code, 
hence they cannot be serialized as naturally 
(and there is a way to do this for compiled models: 
please refer to the [MXNet documentation](http://www.mxnet.io)
for technical details). 
Thus, in order to reinstate a model, we need 
to generate the architecture in code 
and then load the parameters from disk. 
The deferred initialization (:numref:`sec_deferred_init`) 
is advantageous here since we can simply define a model
without the need to put actual values in place. 
Let us start with our familiar MLP.

```{.python .input  n=6}
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))

net = MLP()
net.initialize()
x = np.random.uniform(size=(2, 20))
y = net(x)
```

Next, we store the parameters of the model as a file with the name `mlp.params`.
Gluon Blocks support a `save_parameters` method 
that writes all parameters to disk given 
a string for the file name. 

```{.python .input}
net.save_parameters('mlp.params')
```

To recover the model, we instantiate a clone 
of the original MLP model.
Instead of randomly initializing the model parameters, 
we read the parameters stored in the file directly.
Conveniently we can load parameters into Blocks
via their `load_parameters` method. 

```{.python .input  n=8}
clone = MLP()
clone.load_parameters('mlp.params')
```

Since both instances have the same model parameters, 
the computation result of the same input `x` should be the same. 
Let us verify this.

```{.python .input}
yclone = clone(x)
yclone == y
```

## Summary

* The `save` and `load` functions can be used to perform File I/O for `ndarray` objects.
* The `load_parameters` and `save_parameters` functions allow us to save entire sets of parameters for a network in Gluon.
* Saving the architecture has to be done in code rather than in parameters.

## Exercises

1. Even if there is no need to deploy trained models to a different device, what are the practical benefits of storing model parameters?
1. Assume that we want to reuse only parts of a network to be incorporated into a network of a *different* architecture. How would you go about using, say the first two layers from a previous network in a new network.
1. How would you go about saving network architecture and parameters? What restrictions would you impose on the architecture?

## [Discussions](https://discuss.mxnet.io/t/2329)

![](../img/qr_read-write.svg)
