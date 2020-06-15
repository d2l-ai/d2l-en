# Deferred Initialization
:label:`sec_deferred_init`

So far, it might seem that we got away
with being sloppy in setting up our networks.
Specifically, we did the following unintuitive things,
which might not seem like they should work:

* We defined the network architectures 
  without specifying the input dimensionality.
* We added layers without specifying
  the output dimension of the previous layer.
* We even "initialized" these parameters 
  before providing enough information to determine
  how many parameters our models should contain.

You might be surprised that our code runs at all.
After all, there is no way MXNet 
could tell what the input dimensionality of a network would be.
The trick here is that MXNet *defers initialization*,
waiting until the first time we pass data through the model,
to infer the sizes of each layer *on the fly*.


Later on, when working with convolutional neural networks,
this technique will become even more convenient
since the input dimensionality 
(i.e., the resolution of an image) 
will affect the dimensionality 
of each subsequent layer. 
Hence, the ability to set parameters 
without the need to know,
at the time of writing the code, 
what the dimensionality is 
can greatly simplify the task of specifying 
and subsequently modifying our models. 
Next, we go deeper into the mechanics of initialization.


## Instantiating a Network

To begin, let us instantiate an MLP.

```{.python .input}
from mxnet import init, np, npx
from mxnet.gluon import nn
npx.set_np()

def getnet():
    net = nn.Sequential()
    net.add(nn.Dense(256, activation='relu'))
    net.add(nn.Dense(10))
    return net

net = getnet()
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf


net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])
```

At this point, the network cannot possibly know
the dimensions of the input layer's weights
because the input dimension remains unknown.
Consequently MXNet has not yet initialized any parameters.
We confirm by attempting to access the parameters below.

```{.python .input}
print(net.collect_params)
print(net.collect_params())
```

```{.python .input}
#@tab tensorflow
# Note that each layer objects exist but the weights are empty.
# `net.get_weights()` would through an error since the weights
# have not been initialized yet.
[net.layers[i].get_weights() for i in range(len(net.layers))]
```

Note that while the Parameter objects exist,
the input dimension to each layer is listed as `-1`.
MXNet uses the special value `-1` to indicate
that the parameters dimension remains unknown.
At this point, attempts to access `net[0].weight.data()`
would trigger a runtime error stating that the network
must be initialized before the parameters can be accessed.
Now let us see what happens when we attempt to initialze
parameters via the `initialize` method.

```{.python .input}
net.initialize()
net.collect_params()
```

```{.python .input}
#@tab tensorflow
net.build(input_shape=(2, 20))
net.get_weights()
```

As we can see, nothing has changed. 
When input dimensions are unknown, 
calls to initialize do not truly initalize the parameters.
Instead, this call registers to MXNet that we wish 
(and optionally, according to which distribution)
to initialize the parameters. 
Only once we pass data through the network
will MXNet finally initialize parameters 
and will we see a difference.

```{.python .input}
x = np.random.uniform(size=(2, 20))
net(x)  # Forward computation

net.collect_params()
```

```{.python .input}
#@tab tensorflow
x = tf.random.uniform((2, 20))
net(x)  # Forward computation

net.get_weights()
```

As soon as we know the input dimensionality, 
$\mathbf{x} \in \mathbb{R}^{20}$, 
MXNet can identify the shape of the first layer's weight matrix, 
i.e., $\mathbf{W}_1 \in \mathbb{R}^{256 \times 20}$.
Having recognized the first layer shape, MXNet proceeds
to the second layer, whose dimensionality is $10 \times 256$
and so on through the computational graph
until all shapes are known.
Note that in this case, 
only the first layer requires deferred initialization,
but MXNet initializes sequentially. 
Once all parameter shapes are known, 
MXNet can finally initialize the parameters. 


## Deferred Initialization in Practice

Now that we know how it works in theory, 
let us see when the initialization is actually triggered.
In order to do so, we mock up an initializer 
which does nothing but report a debug message 
stating when it was invoked and with which parameters.

```{.python .input  n=22}
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        # The actual initialization logic is omitted here

net = getnet()
net.initialize(init=MyInit())
```


```{.python .input}
#@tab tensorflow
class MyInit(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        print('Init', shape)
        # For custom tf.keras initializer, the actual
        # initialization logic cannot be omitted here.
        return tf.random.uniform(shape, dtype=dtype)

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=MyInit()),
    tf.keras.layers.Dense(1),
])

# Unlike other framework, this will actually print
# since we already specified the input shape and
# implemented the initialization logic in the
# custom initializer.
net.build(input_shape=(2, 20))
```

Note that, although `MyInit` will print information 
about the model parameters when it is called, 
the above `initialize` function does not print 
any information after it has been executed.  
Therefore there is no real initialization parameter 
when calling the `initialize` function. 
Next, we define the input and perform a forward calculation.

```{.python .input  n=25}
x = np.random.uniform(size=(2, 20))
y = net(x)
```

```{.python .input}
#@tab tensorflow
x = tf.random.uniform((2, 20))
y = net(x)
```

At this time, information on the model parameters is printed. 
When performing a forward calculation based on the input `x`,
the system can automatically infer the shape of the weight parameters 
of all layers based on the shape of the input. 
Once the system has created these parameters, 
it calls the `MyInit` instance to initialize them 
before proceeding to the forward calculation.

This initialization will only be called 
when completing the initial forward calculation. 
After that, we will not re-initialize 
when we run the forward calculation `net(x)`, 
so the output of the `MyInit` instance will not be generated again.

```{.python .input}
y = net(x)
```

```{.python .input}
#@tab tensorflow
y = net(x)
```

As mentioned at the beginning of this section,
deferred initialization can be a source of confusion.
Before the first forward calculation,
we were unable to directly manipulate the model parameters.
For example, we could not use
the `data` and `set_data` functions
to get and modify the parameters.
Therefore, we often force initialization
by sending a sample observation through the network.

## Forced Initialization

Deferred initialization does not occur 
if the system knows the shape of all parameters 
when we call the `initialize` function. 
This can occur in two cases:

* We have already seen some data and we just want to reset the parameters.
* We specified all input and output dimensions of the network when defining it.

Forced reinitialization works as illustrated below.

```{.python .input}
net.initialize(init=MyInit(), force_reinit=True)
```

The second case requires that we specify 
all parameters when creating each layer.
For instance, for dense layers we must specify `in_units` 
at the time that the layer is instantiated.

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(256, in_units=20, activation='relu'))
net.add(nn.Dense(10, in_units=256))

net.initialize(init=MyInit())
```

## Summary

* Deferred initialization can be convenient, allowing Gluon to infer parameter shapes automatically, making it easy to modify architectures and eliminating one common source of errors.
* We do not need deferred initialization when we specify all variables explicitly.
* We can forcibly re-initialize a network's parameters by invoking initalize with the `force_reinit=True` flag.


## Exercises

1. What happens if you specify the input dimensions to the first laye but not to subsequent layers? Do you get immediate initialization?
1. What happens if you specify mismatching dimensions?
1. What would you need to do if you have input of varying dimensionality? Hint - look at parameter tying.

## [Discussions](https://discuss.mxnet.io/t/2327)

![](../img/qr_deferred-init.svg)
