# Deferred Initialization of Model Parameters



If you have done the exercise in the previous section, you will have found that the model `net` had 0 in the shape of the weight parameter after calling the initialization function `initialize` , and before doing the forward calculation of `net(x)`. Although intuitively, `initialize` completes all parameter initialization processes. This, is not necessarily true for Gluon. We will discuss this topic in detail in this section.


## Deferred Initialization

You may have noticed that the fully connected layers created in the previous Gluon did not specify the number of inputs. For example, in the multilayer perceptron `net` used in the previous section, the hidden layer we created only specified an output size of 256. When the `initialize` function is called, since the number of hidden layer inputs is still unknown, the system cannot identify the shape of the layer weight parameter. Only when we pass the input `x` of shape `(2, 20)` into the network for forward calculation `net(x)`, will the system conclude that the weight parameter shape of the layer is `(256, 20)`. Now we can really start to initialize the parameters.

Let's demonstrate this process using the `MyInit` class defined in the previous section. We create a multilayer perceptron and use the `MyInit` instance to initialize the model parameters.

```{.python .input  n=22}
from mxnet import init, nd
from mxnet.gluon import nn
 
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        # The actual initialization logic is omitted here.

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'),
        nn.Dense(10))

net.initialize(init=MyInit())
```

Note that, although `MyInit` will print information about the model parameters when it is called, the above `initialize` function does not print any information after it has been executed.  Therefore there is no real initialization parameter when calling the `initialize` function. Next, we define the input and perform a forward calculation.

```{.python .input  n=25}
x = nd.random.uniform(shape=(2, 20))
y = net(x)
```

At this time, information on the model parameters is printed. When performing a forward calculation based on the input `x`, the system can automatically infer the shape of the weight parameters of all layers based on the shape of the input. Once the system has created these parameters, it calls the `MyInit` instance to initialize them before proceeding to the forward calculation.

Of course, this initialization will only be called when completing the initial forward calculation. After that, we will not re-initialize when we run the forward calculation `net(x)`, so the output of the `MyInit` instance will not be generated again.

```{.python .input}
y = net(x)
```

The act of delaying the actual parameter initialization until it gets enough information is referred to as deferred initialization. It makes the creation of the model simpler: we only need to define the output size of each layer, without manually estimating the number of inputs. This is especially convenient for networks of up containing tens or even hundreds of layers, which we introduced later.

However, everything has two sides. As mentioned at the beginning of this section, deferred initialization can also cause confusion. Before the first forward calculation, we were unable to directly manipulate the model parameters, for example, we could not use the `data` and `set_data` functions to get and modify the parameters. Therefore, we often do an extra forward calculation to force the parameters to be truly initialized.

## Avoid Deferred Initialization

Deferred initialization does not occur if the system knows the shape of all parameters when calling the `initialize` function. We introduce two such situations here.

The first case is when we want to reinitialize the initialized model. Because the shape of the parameter does not change, the system can be reinitialized immediately.

```{.python .input}
net.initialize(init=MyInit(), force_reinit=True)
```

The second case is that we specify the number of inputs when we create the layer, so that the system does not need additional information to speculate the shape of the parameters. In the following example, we specify the number of inputs for each fully connected layer by `in units`, so that initialization can occur immediately when the `initialize` function is called.

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(256, in_units=20, activation='relu'))
net.add(nn.Dense(10, in_units=256))

net.initialize(init=MyInit())
```

## Summary

* The system’s behavior of delaying the actual parameter initialization until information is obtained is referred to as deferred initialization.
* The main benefit of deferred initialization is that it makes the model construction easier. For example, we don't need to manually guess the number of inputs per layer.
* We can also avoid deferred initialization.


## exercise

* What happens if the shape of the input `x` is changed before the next forward calculation of `net(x)`, including the batch size and the number of inputs?

## Scan the QR code to get to the [forum](https://discuss.gluon.ai/t/topic/6320)

![](../img/qr_deferred-init.svg)
