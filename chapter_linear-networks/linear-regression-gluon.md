# Concise Implementation of Linear Regression
:label:`sec_linear_gluon`

Resurgent interest deep learning has inspired the development
of a variety of mature software frameworks,
for automating the repetitive work of implementing
gradient-based learning algorithms.
In the previous section we relied only on
`ndarray` for data storage and linear algebra
and on `autograd` to calculate derivatives.
In practice, because data iterators, loss functions, optimizers,
and neural network layers (and some whole architectures)
are so common, modern libraries implement these components for us as well.

In this section, we will learn how we can implement
the linear regression model in :numref:`sec_linear_scratch` much more concisely with Gluon.

## Generating Data Sets

To start, we will generate the same data set as that used in the previous section.

```{.python .input  n=2}
import d2l
from mxnet import autograd, np, npx, gluon
npx.set_np()

true_w = np.array([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
```

## Reading Data

Rather than rolling our own iterator,
we can call upon Gluon's `data` module to read data.
The first step will be to instantiate an `ArrayDataset`.
This object's constructor takes one or more `ndarray`s as arguments.
Here, we pass in `features` and `labels` as arguments.
Next, we will use the ArrayDataset to instantiate a DataLoader,
which also requires that we specify a `batch_size`
and specify a Boolean value `shuffle` indicating whether or not
we want the `DataLoader` to shuffle the data
on each epoch (pass through the dataset).

```{.python .input  n=3}
# Save to the d2l package. 
def load_array(data_arrays, batch_size, is_train=True):
    """Construct a Gluon data loader"""
    dataset = gluon.data.ArrayDataset(*data_arrays)
    return gluon.data.DataLoader(dataset, batch_size, shuffle=is_train)
    
batch_size = 10
data_iter = load_array((features, labels), batch_size)
```

Now we can use `data_iter` in much the same way as we called 
the `data_iter` function in the previous section. 
To verify that it is working, we can read and print 
the first minibatch of instances.

```{.python .input  n=5}
for X, y in data_iter:
    print(X, '\n', y)
    break
```

## Define the Model

When we implemented linear regression from scratch (in :num_ref`sec_linear_scratch`), 
we defined our model parameters explicitly
and coded up the calculations to produce output 
using basic linear algebra operations. 
You *should* know how to do this.
But once your models get more complex, 
and once you have to do this every day,
you will be glad for the assistance. 

For standard operations, we can use Gluon's predefined layers,
which allow us to focus especially 
on the layers used to construct the model 
rather than having to focus on the implementation.

To define a linear model, we first import the `nn` module,
which defines a large number of neural network layers
(note that "nn" is an abbreviation for neural networks).
We will first define a model variable `net`, 
which is a `Sequential` instance. 
In Gluon, a `Sequential` instance can be regarded as a container
that concatenates the various layers in sequence.
When input data is given, each layer in the container will be calculated in order, 
and the output of one layer will be the input of the next layer.
In this example, our model consists of only one layer,
so we do not really need `Sequential`.
But since nearly all of our future models will involve multiple layers,
let's get into the habit early.

```{.python .input  n=5}
from mxnet.gluon import nn
net = nn.Sequential()
```

Recall the architecture of a single layer network.
The layer is said to be *fully-connected* 
because each of its inputs are connected to each of its outputs 
by means of a matrix-vector multiplication.
In Gluon, the fully-connected layer is defined in the `Dense` class.
Since we only want to generate a single scalar output,
we set that number to $1$.

![Linear regression is a single-layer neural network. ](../img/singleneuron.svg)

```{.python .input  n=6}
net.add(nn.Dense(1))
```

It is worth noting that, for convenience,
Gluon does not require us to specify
the input shape for each layer.
So here, we don't need to tell Gluon
how many inputs go into this linear layer.
When we first try to pass data through our model,
e.g., when we execute `net(X)` later,
Gluon will automatically infer the number of inputs to each layer.
We will describe how this works in more detail
in the chapter "Deep Learning Computation".


## Initialize Model Parameters

Before using `net`, we need to initialize the model parameters,
such as the weights and biases in the linear regression model.
We will import the `initializer` module from MXNet.
This module provides various methods for model parameter initialization.
Gluon makes `init` available as a shortcut (abbreviation)
to access the `initializer` package.
By calling `init.Normal(sigma=0.01)`, 
we specify that each *weight* parameter
should be randomly sampled from a normal distribution
with mean $0$ and standard deviation $0.01$.
The *bias* parameter will be initialized to zero by default. 
Both the weight vector and bias will be attached with gradients.

```{.python .input  n=7}
from mxnet import init
net.initialize(init.Normal(sigma=0.01))
```

The code above looks straightforward but in reality
something quite strange is happening here.
We are initializing parameters for a network
even though Gluon does not yet know 
how many dimensions the input will have!
It might be $2$ as in our example or it might be $2000$.
Gluon lets us get away with this because behind the scenes,
the initialization is actually *deferred*.
The real initialization will take place only
when we for the first time attempt to pass data through the network.
Just be careful to remember that since the parameters
have not been initialized yet,
we cannot access or manipulate them.


## Define the Loss Function

In Gluon, the `loss` module defines various loss functions.
We will the imported module `loss` with the pseudonym `gloss`,
to avoid confusing it for the variable 
holding our chosen loss function.
In this example, we will use the Gluon 
implementation of squared loss (`L2Loss`).

```{.python .input  n=8}
from mxnet.gluon import loss as gloss
loss = gloss.L2Loss()  # The squared loss is also known as the L2 norm loss
```

## Define the Optimization Algorithm

Not surpisingly, we aren't the first people
to implement mini-batch stochastic gradient descent,
and thus `Gluon` supports SGD alongside a number of
variations on this algorithm through its `Trainer` class.
When we instantiate the `Trainer`,
we will specify the parameters to optimize over 
(obtainable from our net via `net.collect_params()`),
the optimization algortihm we wish to use (`sgd`),
and a dictionary of hyper-parameters 
required by our optimization algorithm.
SGD just requires that we set the value `learning_rate`,
(here we set it to 0.03).

```{.python .input  n=9}
from mxnet import gluon
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
```

## Training

You might have noticed that expressing our model through Gluon
requires comparatively few lines of code.
We didn't have to individually allocate parameters,
define our loss function, or implement stochastic gradient descent.
Once we start working with much more complex models,
the benefits of relying on Gluon's abstractions will grow considerably.
But once we have all the basic pieces in place,
the training loop itself is strikingly similar
to what we did when implementing everything from scratch.

To refresh your memory: for some number of epochs,
we'll make a complete pass over the dataset (train_data), 
grabbing one minibatch of inputs 
and corresponding ground-truth labels at a time. 
For each batch, we will go through the following ritual:

* Generate predictions by calling `net(X)` and calculate the loss `l` (the forward pass).
* Calculate gradients by calling `l.backward()` (the backward pass).
* Update the model parameters by invoking our SGD optimizer (note that `trainer` already knows which parameters to optimize over, so we just need to pass in the batch size.

For good measure, we compute the loss after each epoch and print it to monitor progress.

```{.python .input  n=10}
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))
```

The model parameters we have learned 
and the actual model parameters are compared as below. 
We get the layer we need from the `net` 
and access its weight (`weight`) and bias (`bias`). 
The parameters we have learned and the actual parameters are very close.

```{.python .input  n=12}
w = net[0].weight.data()
print('Error in estimating w', true_w.reshape(w.shape) - w)
b = net[0].bias.data()
print('Error in estimating b', true_b - b)
```

## Summary

* Using Gluon, we can implement models much more succinctly.
* In Gluon, the module `data` provides tools for data processing, the module `nn` defines a large number of neural network layers, and the module `loss` defines various loss functions.
* MXNet's module `initializer` provides various methods for model parameter initialization.
* Dimensionality and storage are automagically inferred (but caution if you want to access parameters before they've been initialized).


## Exercises

1. If we replace `l = loss(output, y)` with `l = loss(output, y).mean()`, we need to change `trainer.step(batch_size)` to `trainer.step(1)` accordingly. Why?
1. Review the MXNet documentation to see what loss functions and initialization methods are provided in the modules `gluon.loss` and `init`. Replace the loss by Huber's loss.
1. How do you access the gradient of `dense.weight`?

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2333)

![](../img/qr_linear-regression-gluon.svg)
