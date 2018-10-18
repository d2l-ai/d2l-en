# Custom Layers

One of the charms of deep learning lies in the various layers of the neural network.  Examples include the fully connected layer, the convolutional, pooling, and looping layers, all of which will be introduced in later chapters.   Although Gluon provides a number of commonly used layers, sometimes a custom layer is needed.  This section describes how to use NDArray to customize a Gluon layer, so that it can be repeatedly called. 


## Custom Layer without Model Parameters

We will first show the reader how to define a custom layer that does not contain any model parameters.   In fact, this is similar to using the Block class construction model described in Model Construction section. [ "](model-construction.md). The following `CenteredLayer` class customizes a layer that subtracts the mean from the input before outputting by inheriting the Block class. It defines the calculation of the layer in the `forward` function. There are no model parameters in this layer.

```{.python .input  n=1}
from mxnet import gluon, nd
from mxnet.gluon import nn

class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x - x.mean()
```

We can provide an example for this layer, then do forward calculations. 

```{.python .input  n=2}
layer = CenteredLayer()
layer(nd.array([1, 2, 3, 4, 5]))
```

We can also use it to construct more complex models.

```{.python .input  n=3}
net = nn.Sequential()
net.add(nn.Dense(128),
        CenteredLayer())
```

Next, we print the individual output means of the custom layer. Since the mean is a floating-point number, its value is a number very close to zero.

```{.python .input  n=4}
net.initialize()
y = net(nd.random.uniform(shape=(4, 8)))
y.mean().asscalar()
```

## Custom Layer with Model Parameters

We can also customize the custom layer with model parameters. The model parameters can be taught through training.  

The `Parameter` class and the `ParameterDict` class are introduced in the Access, Initialization, and Sharing of Model Parameters< section [" ](parameters.md). When customizing the layer containing model parameters, we can use the member variable `params` of the `ParameterDict` type that comes with the Block class. It is a dictionary that maps string type parameter names to model parameters in the Parameter type.  We can create a `Parameter` instance from `ParameterDict` via the `get` function.

```{.python .input  n=7}
params = gluon.ParameterDict()
params.get('param2', shape=(2, 3))
params
```

Next, we try to implement a fully connected layer with both weight and bias parameters.  It uses ReLU as an activation function, where `in_units` and `units` are the number of inputs and the number of outputs, respectively.

```{.python .input  n=19}
class MyDense(nn.Block):
    # Units: the number of outputs in this layer; in_units: the number of inputs in this layer.
    def __init__(self, units, in_units, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        linear = nd.dot(x, self.weight.data()) + self.bias.data()
        return nd.relu(linear)
```

Next, we instantiate the `MyDense` class and access its model parameters.

```{.python .input}
dense = MyDense(units=3, in_units=5)
dense.params
```

We can directly carry out forward calculations using custom layers. 

```{.python .input  n=20}
dense.initialize()
dense(nd.random.uniform(shape=(2, 5)))
```

We can also construct models using custom layers. It is similar in use to other layers of Gluon.

```{.python .input  n=19}
net = nn.Sequential()
net.add(MyDense(8, in_units=64),
        MyDense(1, in_units=8))
net.initialize()
net(nd.random.uniform(shape=(2, 64)))
```

## Summary

* We can customize the layers in the neural network through the Block class so that they can be repeatedly called. 


## exercise

* Customize a layer and use it to do a forward calculation.


## Scan the QR code to get to the [ ](https://discuss.gluon.ai/t/topic/1256) forum

![](../img/qr_custom-layer.svg)
