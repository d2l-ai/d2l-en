# Concise Implementation of Linear Regression

The surge of deep learning has inspired the development 
of a variety of mature software frameworks,
that automate much of the repetitive work 
of implementing deep learning models.
In the previous section we relied only 
on NDarray for data storage and linear algebra
and the auto-differentiation capabilities in the `autograd` package.
In practice, because many of the more abstract operations, e.g.
data iterators, loss functions, model architectures, and optimizers,
are so common, deep learning libraries will give us 
library functions for these as well. 

In this section, we will introduce Gluon, MXNet's high-level interface
for implementing neural networks and show how we can implement 
the linear regression model from the previous section much more concisely.

## Generating Data Sets

To start, we will generate the same data set as that used in the previous section.

```{.python .input  n=2}
from mxnet import autograd, nd

num_inputs = 2
num_examples = 1000
true_w = nd.array([2, -3.4])
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = nd.dot(features, true_w) + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)
```

## Reading Data

Rather than rolling our own iterator, 
we can call upon Gluon's `data` module to read data. 
Since `data` is often used as a variable name, 
we will replace it with the pseudonym `gdata` 
(adding the first letter of Gluon),
too differentiate the imported `data` module
from a variable we might define. 
The first step will be to instantiate an `ArrayDataset`,
which takes in one or more NDArrays as arguments.
Here, we pass in `features` and `labels` as arguments.
Next, we will use the ArrayDataset to instantiate a DataLoader,
which also requires that we specify a `batch_size` 
and specify a Boolean value `shuffle` indicating whether or not 
we want the `DataLoader` to shuffle the data 
on each epoch (pass through the dataset).

```{.python .input  n=3}
from mxnet.gluon import data as gdata

batch_size = 10
# Combine the features and labels of the training data
dataset = gdata.ArrayDataset(features, labels)
# Randomly reading mini-batches
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
```

Now we can use `data_iter` in much the same way as we called the `data_iter` function in the previous section. To verify that it's working, we can read and print the first mini-batch of instances.

```{.python .input  n=5}
for X, y in data_iter:
    print(X, y)
    break
```

## Define the Model

When we implemented linear regression from scratch in the previous section, we had to define the model parameters and explicitly write out the calculation to produce output using basic linear algebra opertions. You should know how to do this. But once your models get more complex, even qualitatively simple changes to the model might result in many low-level changes.

For standard operations, we can use Gluon's predefined layers, which allow us to focus especially on the layers used to construct the model rather than having to focus on the implementation.

To define a linear model, we first import the `nn` module,
which defines a large number of neural network layers
(note that "nn" is an abbreviation for neural networks). 
We will first define a model variable `net`, which is a `Sequential` instance. In Gluon, a `Sequential` instance can be regarded as a container 
that concatenates the various layers in sequence. 
When input data is given, each layer in the container will be calculated in order, and the output of one layer will be the input of the next layer.
In this example, since our model consists of only one layer,
we do not really need `Sequential`.
But since nearly all of our future models will involve multiple layers, 
let's get into the habit early.


```{.python .input  n=5}
from mxnet.gluon import nn
net = nn.Sequential()
```

Recall the architecture of a single layer network. 
The layer is fully connected since it connects all inputs 
with all outputs by means of a matrix-vector multiplication. 
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
e.g., when we exedcute `net(X)` later, 
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
By calling `init.Normal(sigma=0.01)`, we specify that each *weight* parameter
should be randomly sampled from a normal distribution 
with mean 0 and standard deviation 0.01. 
The *bias* parameter will be initialized to zero by default.

```{.python .input  n=7}
from mxnet import init
net.initialize(init.Normal(sigma=0.01))
```

The code above looks straightforward but in reality 
something quite strange is happening here. 
We are initializing parameters for a network
even though we haven't yet told Gluon how many dimensions the input will have. 
It might be 2 as in our example or it might be 2,000, 
so we couldn't just preallocate enough space to make it work.

Gluon let's us get away with this because behind the scenes,
the initialization is deferred until the first time 
that we attempt to pass data through our network. 
Just be careful to remember that since the parameters 
have not been initialized yet we cannot yet manipulate them in any way.


## Define the Loss Function

In Gluon, the `loss` module defines various loss functions. 
We will replace the imported module `loss` with the pseudonym `gloss`, 
and directly use its implementation of squared loss (`L2Loss`).

```{.python .input  n=8}
from mxnet.gluon import loss as gloss
loss = gloss.L2Loss()  # The squared loss is also known as the L2 norm loss
```

## Define the Optimization Algorithm

Not surpisingly, we aren't the first people 
to implement mini-batch stochastic gradient descent,
and thus `Gluon` supports SGD alongside a number of 
variations on this algorithm through its `Trainer` class. 
When we instantiate the `Trainer`, we'll specify the parameters to optimize over (obtainable from our net via `net.collect_params()`),
the optimization algortihm we wish to use (`sgd`),
and a dictionary of hyper-parameters required by our optimization algorithm.
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
grabbing one mini-batch of inputs and corresponding ground-truth labels at a time. For each batch, we'll go through the following ritual:

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

The model parameters we have learned and the actual model parameters are compared as below. We get the layer we need from the `net` and access its weight (`weight`) and bias (`bias`). The parameters we have learned and the actual parameters are very close.

```{.python .input  n=12}
w = net[0].weight.data()
print('Error in estimating w', true_w.reshape(w.shape) - w)
b = net[0].bias.data()
print('Error in estimating b', true_b - b)
```

## Summary

* Using Gluon, we can implement the model more succinctly.
* In Gluon, the module `data` provides tools for data processing, the module `nn` defines a large number of neural network layers, and the module `loss` defines various loss functions.
* MXNet's module `initializer` provides various methods for model parameter initialization.
* Dimensionality and storage are automagically inferred (but caution if you want to access parameters before they've been initialized).


## Exercises

1. If we replace `l = loss(output, y)` with `l = loss(output, y).mean()`, we need to change `trainer.step(batch_size)` to `trainer.step(1)` accordingly. Why?
1. Review the MXNet documentation to see what loss functions and initialization methods are provided in the modules `gluon.loss` and `init`. Replace the loss by Huber's loss.
1. How do you access the gradient of `dense.weight`?

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2333)

![](../img/qr_linear-regression-gluon.svg)
