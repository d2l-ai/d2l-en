# Reading and Storage

So far, we have introduced how to process data and build, train, and test deep learning models. In practice, however, we sometimes need to deploy trained models to many different devices. In this case, we can store the model parameters trained in memory on the hard disk for subsequent reading.


## Read and Write NDArrays

We can directly use the `save` and `load` functions to store and read NDArrays separately. The following example creates the NDArray variable `x` and stores it in a file with the same name as `x`.

```{.python .input}
from mxnet import nd
from mxnet.gluon import nn

x = nd.ones(3)
nd.save('x', x)
```

Then, we read the data from the stored file back into memory.

```{.python .input}
x2 = nd.load('x')
x2
```

We can also store a list of NDArrays and read them back into memory.

```{.python .input  n=2}
y = nd.zeros(4)
nd.save('xy', [x, y])
x2, y2 = nd.load('xy')
(x2, y2)
```

We can even store and read a dictionary that maps from a string to an NDArray.

```{.python .input  n=4}
mydict = {'x': x, 'y': y}
nd.save('mydict', mydict)
mydict2 = nd.load('mydict')
mydict2
```

## Read and Write Gluon Model Parameters

In addition to NDArray, we can also read and write parameters of the Gluon model. Gluon's Block class provides the `save_parameters` and `load_parameters` functions to read and write model parameters. For the sake of demonstration, we will first create a multilayer perceptron and initialize it. Recall the ["Deferred Initialization of Model Parameters" ](deferred- Init.md) section, due to deferred initialization, we need to run a forward computation first to actually initialize the model parameters.

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
x = nd.random.uniform(shape=(2, 20))
y = net(x)
```

Next, we will store the parameters of the model as a file with the file name as "mlp.params".

```{.python .input}
filename = 'mlp.params'
net.save_parameters(filename)
```

Next, we instantiate a defined multilayer perceptron. Unlike the random initialization of model parameters, here we read the parameters stored in the file directly.

```{.python .input  n=8}
net2 = MLP()
net2.load_parameters(filename)
```

Since both instances have the same model parameters, the computation result of the same input `x` will be the same. Now, we will verify it.

```{.python .input}
y2 = net2(x)
y2 == y
```

## Summary

* By using the `save` and `load` functions, it is easy to read and write NDArray.
* By using the `load_parameters` and `save_parameters` functions, it is easy to read and write parameters of the Gluon model.

## exercise

* Even if there is no need to deploy trained models to a different device, what are the practical benefits of storing model parameters?

## Scan the QR Code to Access [Discussions](https://discuss.gluon.ai/t/topic/1255)

![](../img/qr_read-write.svg)
