# Gluon Implementation of Linear Regression

With the development of deep learning frameworks, it has become increasingly easy to develop deep learning applications. In practice, we can usually implement the same model, but with a more concise code than that introduced in the previous section. In this section, we will introduce how to use the Gluon interface provided by MXNet to more easily implement linear regression training.

## Generating Data Sets

We will generate the same data set as that used in the previous section,  where `features` is the feature of training data, and `labels` is the label.

```{.python .input  n=2}
from mxnet import autograd, nd

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)
```

## Reading Data

Gluon provides the `data` module to read data. Since `data` is often used as a variable name, we will replace it with the pseudonym `gdata` (adding the first letter of Gluon) when referring to the imported `data` module. In each iteration, we will randomly read a mini-batch containing 10 data instances.

```{.python .input  n=3}
from mxnet.gluon import data as gdata

batch_size = 10
# Combining the features and labels of the training data.
dataset = gdata.ArrayDataset(features, labels)
# Randomly reading mini-batches.
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
```

The use of `data_iter` here is the same as in the previous section. Now, we can read and print the first mini-batch of data instances.

```{.python .input  n=5}
for X, y in data_iter:
    print(X, y)
    break
```

## Define the Model

In the implementation of linear regression starting from scratch in the previous section, we needed to define the model parameters and use them to describe how the model is calculated step by step. These steps become more complicated as the model structure becomes more complex. In fact, Gluon provides a large number of predefined layers, which allow us to focus especially on the layers used to construct the model. Next, we will introduce how to use Gluon to define linear regression more succinctly.

First, import the module `nn`. "nn" is an abbreviation of neural networks. As the name implies, this module defines a large number of neural network layers. We will first define a model variable `net`, which is a Sequential instance. In Gluon, a Sequential instance can be regarded as a container that concatenates the various layers. When constructing the model, we will add the layers in order in the container. When input data is given, each layer in the container will be calculated in order, and the output will be the input of the next layer.

```{.python .input  n=5}
from mxnet.gluon import nn

net = nn.Sequential()
```

Review the representation of linear regression in the neural network diagram in Figure 3.1. As a single-layer neural network, the neurons in the linear regression output layer are fully connected to the inputs in the input layer. Therefore, the output layer of linear regression is also called the fully connected layer. In Gluon, the fully connected layer is a `Dense` instance. Here, we define the number of outputs for this layer as 1.

```{.python .input  n=6}
net.add(nn.Dense(1))
```

It is worth noting that, in Gluon, we do not need to specify the input shape for each layer, such as the number of linear regression inputs. When the model sees the data, for example, when the `net(X)` is executed later, the model will automatically infer the number of inputs in each layer. We will describe this mechanism in detail in the chapter "Deep Learning Computation".   Gluon introduces this design to make model development more convenient.


## Initialize Model Parameters

Before using `net`, we need to initialize the model parameters, such as the weights and biases in the linear regression model. We will import the `initializer` module from MXNet. This module provides various methods for model parameter initialization. The `init` here is the abbreviation of `initializer`. By`init.Normal(sigma=0.01)` we specify that each weight parameter element is to be randomly sampled at initialization with a normal distribution with a mean of 0 and standard deviation of 0.01. The bias parameter will be initialized to zero by default.

```{.python .input  n=7}
from mxnet import init

net.initialize(init.Normal(sigma=0.01))
```

## Define Loss Function

In Gluon, the module `loss` defines various loss functions. We will replace the imported module `loss` with the pseudonym `gloss`, and directly use the squared loss it provides as a loss function for the model.

```{.python .input  n=8}
from mxnet.gluon import loss as gloss

loss = gloss.L2Loss()  # The squared loss is also known as the L2 norm loss.
```

## Define the Optimization Algorithm

Again, we do not need to implement mini-batch stochastic gradient descent. After importing Gluon, we now create a `Trainer` instance and specify a mini-batch stochastic gradient descent with a learning rate of 0.03 (`sgd`) as the optimization algorithm. This optimization algorithm will be used to iterate through all the parameters contained in the `net` instance's nested layers through the `add` function.  These parameters can be obtained by the `collect_params` function.

```{.python .input  n=9}
from mxnet import gluon

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
```

## To train a model

When using the Gluon training model, we update the module parameters by calling the `step` function of the `Trainer` instance. As we introduced in the previous section, since the variable `l` is a one-dimensional NDArray with a length of `batch_size`, executing `l.backward()` is equivalent to executing `l.sum().backward()`. According to the definition of mini-batch stochastic gradient descent, we specify the batch size in the `step` function to calculate the average of example Gradient the in the batch.

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
dense = net[0]
true_w, dense.weight.data()
```

```{.python .input  n=13}
true_b, dense.bias.data()
```

## Summary

* Using Gluon, we can implement the model more succinctly.
* In Gluon, the module `data` provides tools for data processing, the module `nn` defines a large number of neural network layers, and the module `loss` defines various loss functions.
* MXNet's module `initializer` provides various methods for model parameter initialization.


## exercise

* If we replace `l = loss(output, y)` with `l = loss(output, y).mean()`, we need to change `trainer.step(batch_size)` to `trainer.step(1)` accordingly. Why?
* Review the MXNet documentation to see what loss functions and initialization methods are provided in the modules `gluon.loss` and `init`.
* How do you access the gradient of `dense.weight`?


## Scan the QR Code to Access [Discussions](https://discuss.gluon.ai/t/topic/742)

![](../img/qr_linear-regression-gluon.svg)
