# Layers and Blocks

One of the key components that helped propel deep learning is powerful software. In an analogous manner to semiconductor design where engineers went from specifying transistors to logical circuits to writing code we now witness similar progress in the design of deep networks. The previous chapters have seen us move from designing single neurons to entire layers of neurons. However, even network design by layers can be tedious when we have 152 layers, as is the case in ResNet-152, which was proposed by [He et al.](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) in 2016 for computer vision problems. 
Such networks have a fair degree of regularity and they consist of *blocks* of repeated (or at least similarly designed) layers. These blocks then form the basis of more complex network designs. In short, blocks are combinations of one or more layers. This design is aided by code that generates such blocks on demand, just like a Lego factory generates blocks which can be combined to produce terrific artifacts. 

To get started we design a very simple block, namely the block for a multilayer perceptron, such as the one we encountered [previously](../chapter_deep-learning-basics/mlp-gluon.md). A common strategy would be to design a two-layer network as follows:

```{.python .input  n=1}
from mxnet import nd
from mxnet.gluon import nn

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
```

This generates a network with a hidden layer of 256 units, followed by a ReLu activation and another 10 units governing the output. In particular, we used the `nn.Sequential` constructor to generate an empty network into which we then inserted both layers. 


section. First, we construct a Sequential instance, and then add two fully connected layers, one at a time. The output size of the first layer is 256, so the number of hidden layer units is 256. The output size of the second layer is 10, so the number of output layer units is 10. We also used the Sequential class construction model in the other sections of the previous chapter. Here, we introduce another model construction method based on the Block class, makes model construction more flexible.


## Inherit the Block Class to Construct the Model

The Block class is a model constructor provided in the `nn` module, which we can inherit to define the model we want. The following inherits the Block class to construct the multilayer perceptron mentioned at the beginning of this section. The `MLP` class defined here overrides the `__init__` and `forward` functions of the Block class. They are used to create model parameters and define forward computations, respectively. Forward computation is also forward propagation.

```{.python .input  n=1}
from mxnet import nd
from mxnet.gluon import nn

class MLP(nn.Block):
    # Declare a layer with model parameters. Here, we declare two fully connected layers.
    def __init__(self, **kwargs):
        # Call the constructor of the MLP parent class Block to perform the necessary initialization. In this way,
        # other function parameters can also be specified when constructing an instance, such as the model parameter, params, described in the following sections.
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')  # Hidden layer.
        self.output = nn.Dense(10)  # Output layer.

    # Define the forward computation of the model, that is, how to return the required model output based on the input x.
    def forward(self, x):
        return self.output(self.hidden(x))
```

There is no need to define a back propagation function in the above `MLP` class. The system automatically generates the `backward` function needed for back propagation by automatically finding the gradient.

We can instantiate the `MLP` class to get the model variable `net`. The following code initializes `net` and passes in the input data `x` to perform a forward computation. Where `net(x)` will call the `__call__` function inherited from the Block class by `MLP`, this function will call `forward` the function defined by the `MLP` class to complete forward computation.

```{.python .input  n=2}
x = nd.random.uniform(shape=(2, 20))
net = MLP()
net.initialize()
net(x)
```

Note that we did not give the Block class a name such as Layer or Model because the class is a component that can be freely constructed. Its subclass can either be a layer (such as the `Dense` class provided by Gluon), a model (for example, the MLP class defined here), or a part of a model. We will show its flexibility through the two examples below.

## Sequential Class Inherited from the Block Class

As we just mentioned, the Block class is a generic component. In fact, the Sequential class is inherited from the Block class. When the forward computation of the model is a simple concatenation of computations for each layer, we can define the model in a much simpler way. The purpose of the Sequential class is to provide the `add` function to add concatenated Block subclass instances one by one, while the forward computation of the model is to compute these instances one by one in the order of addition.

Below, we implement a `MySequential` class that has the same functionality as the Sequential class. This may help you understand more clearly how the Sequential class works.

```{.python .input  n=3}
class MySequential(nn.Block):
    def __init__(self, **kwargs):
        super(MySequential, self).__init__(**kwargs)

    def add(self, block):
        # Here, block is an instance of a Block subclass, and we assume it has a unique name. We save it in 
        # the member variable _children of the Block class, and its type is OrderedDict. When the MySequential instance calls the
        # initialize function, the system automatically initializes all members of _children.
        self._children[block.name] = block

    def forward(self, x):
        # OrderedDict guarantees that members will be traversed in the order they were added.
        for block in self._children.values():
            x = block(x)
        return x
```

We use the MySequential class to implement the `MLP` class described earlier and perform a forward computation using a randomly initialized model.

```{.python .input  n=4}
net = MySequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
net(x)
```

It can observed here that the use of the `MySequential` class is no different from the use of the Sequential class described in the [“Gluon implementation of multilayer perceptron”](../chapter_deep-learning-basics/mlp-gluon.md) section.


## Constructing Complex Models

Although the Sequential class can make model construction easier, and you do not need to define the `forward` function, directly inheriting the Block class can greatly expand the flexibility of model construction. Next, we construct a slightly more complex network `FancyMLP`. In this network, we use the `get_constant` function to create parameters that are not iterated during training, i.e. constant parameters. In forward computation, in addition to using the created constant parameters, we also use the NDArray function and Python's control flow, and call the same layer multiple times.

```{.python .input  n=5}
class FancyMLP(nn.Block):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        # Random weight parameters created with the get_constant are not iterated during training (i.e. constant parameters).
        self.rand_weight = self.params.get_constant(
            'rand_weight', nd.random.uniform(shape=(20, 20)))
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, x):
        x = self.dense(x)
        # Use the constant parameters created, as well as the relu and dot functions of NDArray.
        x = nd.relu(nd.dot(x, self.rand_weight.data()) + 1)
        # Reuse the fully connected layer. This is equivalent to sharing parameters with two fully connected layers.
        x = self.dense(x)
        # Here in Control flow, we need to call asscalar to return the scalar for comparison.
        while x.norm().asscalar() > 1:
            x /= 2
        if x.norm().asscalar() < 0.8:
            x *= 10
        return x.sum()
```

In this `FancyMLP` model, we used constant weight `Rand_weight` (note that it is not a model parameter), performed a matrix multiplication operation (`nd.dot<`), and reused the same `Dense` layer. Let us test the random initialization and forward computation of the model.

```{.python .input  n=6}
net = FancyMLP()
net.initialize()
net(x)
```

Since `FancyMLP` and Sequential classes are both subclasses of the Block class, we can call them nested.

```{.python .input  n=7}
class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation='relu'),
                     nn.Dense(32, activation='relu'))
        self.dense = nn.Dense(16, activation='relu')

    def forward(self, x):
        return self.dense(self.net(x))

net = nn.Sequential()
net.add(NestMLP(), nn.Dense(20), FancyMLP())

net.initialize()
net(x)
```

## Summary

* We can construct the model by inheriting the Block class.
* The Sequential class is inherited from the Block class.
* Although the Sequential class can make model construction simpler, directly inheriting the Block class can greatly expand the flexibility of model construction.


## exercise

* What kind of error message will occur when calling an `__init__` function whose parent class not in the `__init__` function of the `MLP` class?
* What kinds of problems will occur if you remove the `asscalar` function in the `FancyMLP` class?
* What kinds of problems will occur if you change `self.net` defined by the Sequential instance in the `NestMLP` class to `self.net = [nn.Dense(64, activation='relu'), nn. Dense(32, activation='relu')]`?


## Scan the QR Code to Access [Discussions](https://discuss.gluon.ai/t/topic/986)


![](../img/qr_model-construction.svg)
