# Access, Initialization, and Sharing of Model Parameters

In the previous chapters, we have always used default methods to initialize all the model’s parameters. We also introduced a simple way to access model parameters. This section provides an in-depth explanation of how to access and initialize model parameters and how to share the same model parameters across multiple layers.

We first define a multilayer perceptron containing a single hidden layer, similar to that in the previous section . We still initialize its parameters in the default way and perform one forward computation. Unlike before, here, we import the `init` package from MXNet, which contains a variety of model initialization methods.

```{.python .input  n=1}
from mxnet import init, nd
from mxnet.gluon import nn

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()  # Use the default initialization method.

x = nd.random.uniform(shape=(2, 20))
y = net(x)  # Forward computation.
```

## Access Model Parameters

For neural networks constructed using the Sequential class, we can access any layer of the network by using square brackets `[]`. Recall the inheritance relationship between the Sequential class and the Block class mentioned in the previous section. For the layer containing the model parameters in the Sequential instance, we can access all parameters contained in this layer by the Block classes’ `params` property.  Next, access all the parameters of the hidden layer in the multilayer perceptron `net`. The index 0 indicates that the hidden layer is the layer added first for the Sequential instance.

```{.python .input  n=2}
net[0].params, type(net[0].params)
```

As you can see, we get a dictionary that maps parameter names to parameter instances (of type `ParameterDict`). The name of the weight parameter is `dense0_weight`, which is composed of `net[0]`'s name (`dense0_`) and its own variable name (`weight`). You can also see that the shape of this parameter is `(256,20)`, and the data type is a 32-bit floating point number (`float32`). In order to access a specific parameter, we can either access the element in the dictionary by name or by directly using its variable name. The following two methods are equivalent, but the latter usually has better code readability.

```{.python .input  n=3}
net[0].params['dense0_weight'], net[0].weight
```

The parameter type in Gluon is the Parameter class, which contains the values​ of the parameters and gradients, which can be accessed by `data` and `grad` functions, respectively. Because we randomly initialized the weights, the weight parameter is an NDArray that consists of random numbers and has a shape of `(256, 20)`.

```{.python .input  n=4}
net[0].weight.data()
```

The shape of the gradient is the same as the weight. Since we have not performed back propagation computation, the values ​​of the gradients are all 0.

```{.python .input  n=5}
net[0].weight.grad()
```

Similarly, we can access the parameters of other layers, such as the bias values of the output layer.

```{.python .input  n=6}
net[1].bias.data()
```

Finally, we can use the `collect_params` function to get all the parameters contained in all the nested layers of the `net` variable (for example, nested by the `add` function). It also returns a dictionary from the parameter name to the parameter instance.

```{.python .input  n=7}
net.collect_params()
```

This function can match the parameter name by a regular expression in order to filter the required parameters.

```{.python .input  n=8}
net.collect_params('.*weight')
```

## Initialize Model Parameters

We described the default initialization method for the model in the [“Numeric Stability and Model Initialization”](../chapter_deep-learning-basics/numerical-stability-and-init.md) section: the weight parameter element is a uniformly distributed random number between [-0.07, 0.07], and the bias parameters are all 0. However, we often need to use other methods to initialize the weights. MXNet's `init` module provides a variety of preset initialization methods. In the following example, we initialize the weight parameters to a normally distributed random number with a mean of 0 and a standard deviation of 0.01, but we still reset the bias parameter to zero.

```{.python .input  n=9}
# Non-first initialization of the model requires that force_reinit be true.
net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
net[0].weight.data()[0]
```

The following use constants to initialize the weight parameters.

```{.python .input  n=10}
net.initialize(init=init.Constant(1), force_reinit=True)
net[0].weight.data()[0]
```

If you want to initialize only a specific parameter, then we can call the `initialize` function of the `Parameter` class, which is consistent with the use of the `initialize` function provided by the Block class. In the following example, we use the Xavier initialization method for the weight of the hidden layer.

```{.python .input  n=11}
net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
net[0].weight.data()[0]
```

## Custom Initialization Method

Sometimes, the initialization methods we need are not provided in the `init` module. At this point, we can implement a subclass of the `Initializer` class so that we can use it like any other initialization method. Usually, we only need to implement the `_init_weight` function and modify the incoming NDArray according to the initial result. In the example below, we have half the probability that the weight is initialized to 0. The other half of the probability is initialized to evenly distributed random numbers in two intervals of $[-10,-5]$ and $[5,10]$.

```{.python .input  n=12}
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = nd.random.uniform(low=-10, high=10, shape=data.shape)
        data *= data.abs() >= 5

net.initialize(MyInit(), force_reinit=True)
net[0].weight.data()[0]
```

In addition, we can also directly overwrite the model parameters by the `set_data` function of the `Parameter` class. In the following example, we add 1 to the existing hidden layer parameters.

```{.python .input  n=13}
net[0].weight.set_data(net[0].weight.data() + 1)
net[0].weight.data()[0]
```

## Share Model Parameters

In some cases, we want to share model parameters across multiple layers. [The "Model Construction" ](model-construction.md) section describes how to call the same layer to compute in the `forward` function of the Block class. Here is another method that uses specific parameters when constructing layers. If different layers use the same parameter, they share the same parameters for both forward and back propagation. In the example below, we let the model's second hidden layer (`shared` variable) and the third hidden layer share the model parameter.

```{.python .input  n=14}
net = nn.Sequential()
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()

x = nd.random.uniform(shape=(2, 20))
net(x)

net[1].weight.data()[0] == net[2].weight.data()[0]
```

We specify that the third hidden layer uses the parameters of the second hidden layer by using `params` when constructing the third hidden layer. Since the model parameters contain gradients, the gradients of the second hidden layer and the third hidden layer are accumulated in the `shared.params.grad( )` during back propagation computation.


## Summary

* We have several ways to access, initialize, and share model parameters.
* We can customize the initialization method.


## exercise

* Refer to the MXNet documentation regarding the `init` module for different parameter initialization methods.
* Try accessing the model parameters after`net.initialize()` and before `net(x)` to observe the shape of the model parameters.
* Construct a multilayer perceptron containing a shared parameter layer and train it. During the training process, observe the model parameters and gradients of each layer.

## Scan the QR Code to Access [Discussions](https://discuss.gluon.ai/t/topic/987)

![](../img/qr_parameters.svg)
