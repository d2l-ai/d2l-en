# Parameter Management

The ultimate goal of training deep networks is to find good parameter values for a given architecture. When everything is standard, the `nn.Sequential` class is a perfectly good tool for it. However, very few models are entirely standard and most scientists want to build things that are novel. This section shows how to manipulate parameters. In particular we will cover the following aspects:

* Accessing parameters for debugging, diagnostics, to visualize them or to save them is the first step to understanding how to work with custom models. 
* Secondly, we want to set them in specific ways, e.g. for initialization purposes. We discuss the structure of parameter initializers.
* Lastly, we show how this knowledge can be put to good use by building networks that share some parameters. 

As always, we start from our trusty Multilayer Perceptron with a hidden layer. This will serve as our choice for demonstrating the various features.

```{.python .input  n=1}
from mxnet import init, nd
from mxnet.gluon import nn

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()  # Use the default initialization method.

x = nd.random.uniform(shape=(2, 20))
net(x)            # Forward computation.
```

```{.json .output n=1}
[
 {
  "data": {
   "text/plain": "\n[[ 0.09543003  0.04614332 -0.00286653 -0.07790346 -0.05130243  0.02942039\n   0.08696645 -0.0190793  -0.04122177  0.05088576]\n [ 0.0769287   0.03099705  0.00856576 -0.04467198 -0.0692684   0.09132432\n   0.06786594 -0.06187843 -0.03436674  0.04234695]]\n<NDArray 2x10 @cpu(0)>"
  },
  "execution_count": 1,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## Parameter Access

In the case of a Sequential class we can access the parameters with ease, simply by indexing each of the layers in the network. The params variable then contains the required data. Let's try this out in practice by inspecting the parameters of the first layer.

```{.python .input  n=2}
print(net[0].params)
print(net[1].params)
```

```{.json .output n=2}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "dense0_ (\n  Parameter dense0_weight (shape=(256, 20), dtype=float32)\n  Parameter dense0_bias (shape=(256,), dtype=float32)\n)\ndense1_ (\n  Parameter dense1_weight (shape=(10, 256), dtype=float32)\n  Parameter dense1_bias (shape=(10,), dtype=float32)\n)\n"
 }
]
```

The output tells us a number of things. Firstly, the layer consists of two sets of parameters: `dense0_weight` and `dense0_bias`, as we would expect. They are both single precision and they have the necessary shapes that we would expect from the first layer, given that the input dimension is 20 and the output dimension 256. In particular the names of the parameters are very useful since they allow us to identify parameters *uniquely* even in a network of hundreds of layers and with nontrivial structure. The second layer is structured accordingly. 

### Targeted Parameters

In order to do something useful with the parameters we need to access them, though. There are several ways to do this, ranging from simple to general. Let's look at some of them. 

```{.python .input  n=3}
print(net[1].bias)
print(net[1].bias.data())
```

```{.json .output n=3}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Parameter dense1_bias (shape=(10,), dtype=float32)\n\n[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n<NDArray 10 @cpu(0)>\n"
 }
]
```

The first returns the bias of the second layer. Sine this is an object containing data, gradients, and additional information, we need to request the data explicitly. Note that the bias is all 0 since we initialized the bias to contain all zeros. Note that we can also access the parameters by name, such as `dense0_weight`. This is possible since each layer comes with its own parameter dictionary that can be accessed directly. Both methods are entirely equivalent but the first method leads to much more readable code. 

```{.python .input  n=4}
print(net[0].params['dense0_weight'])
print(net[0].params['dense0_weight'].data())
```

```{.json .output n=4}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Parameter dense0_weight (shape=(256, 20), dtype=float32)\n\n[[ 0.06700657 -0.00369488  0.0418822  ... -0.05517294 -0.01194733\n  -0.00369594]\n [-0.03296221 -0.04391347  0.03839272 ...  0.05636378  0.02545484\n  -0.007007  ]\n [-0.0196689   0.01582889 -0.00881553 ...  0.01509629 -0.01908049\n  -0.02449339]\n ...\n [ 0.00010955  0.0439323  -0.04911506 ...  0.06975312  0.0449558\n  -0.03283203]\n [ 0.04106557  0.05671307 -0.00066976 ...  0.06387014 -0.01292654\n   0.00974177]\n [ 0.00297424 -0.0281784  -0.06881659 ... -0.04047417  0.00457048\n   0.05696651]]\n<NDArray 256x20 @cpu(0)>\n"
 }
]
```

Note that the weights are nonzero. This is by design since they were randomly initialized when we constructed the network. `data` is not the only function that we can invoke. For instance, we can compute the gradient with respect to the parameters. It has the same shape as the weight. However, since we did not invoke backpropagation yet, the values are all 0.

```{.python .input  n=5}
net[0].weight.grad()
```

```{.json .output n=5}
[
 {
  "data": {
   "text/plain": "\n[[0. 0. 0. ... 0. 0. 0.]\n [0. 0. 0. ... 0. 0. 0.]\n [0. 0. 0. ... 0. 0. 0.]\n ...\n [0. 0. 0. ... 0. 0. 0.]\n [0. 0. 0. ... 0. 0. 0.]\n [0. 0. 0. ... 0. 0. 0.]]\n<NDArray 256x20 @cpu(0)>"
  },
  "execution_count": 5,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

### All Parameters at Once

Accessing parameters as described above can be a bit tedious, in particular if we have more complex blocks, or blocks of blocks (or even blocks of blocks of blocks), since we need to walk through the entire tree in reverse order to how the blocks were constructed. To avoid this, blocks come with a method `collect_params` which grabs all parameters of a network in one dictionary such that we can traverse it with ease. It does so by iterating over all constituents of a block and calls `collect_params` on subblocks as needed. To see the difference consider the following:

```{.python .input  n=6}
# parameters only for the first layer
print(net[0].collect_params())
# parameters of the entire network
print(net.collect_params())
```

```{.json .output n=6}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "dense0_ (\n  Parameter dense0_weight (shape=(256, 20), dtype=float32)\n  Parameter dense0_bias (shape=(256,), dtype=float32)\n)\nsequential0_ (\n  Parameter dense0_weight (shape=(256, 20), dtype=float32)\n  Parameter dense0_bias (shape=(256,), dtype=float32)\n  Parameter dense1_weight (shape=(10, 256), dtype=float32)\n  Parameter dense1_bias (shape=(10,), dtype=float32)\n)\n"
 }
]
```

This provides us with a third way of accessing the parameters of the network. If we wanted to get the value of the bias term of the second layer we could simply use this:

```{.python .input  n=7}
net.collect_params()['dense1_bias'].data()
```

```{.json .output n=7}
[
 {
  "data": {
   "text/plain": "\n[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n<NDArray 10 @cpu(0)>"
  },
  "execution_count": 7,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Throughout the book we'll see how various blocks name their subblocks (Sequential simply numbers them). This makes it very convenient to use regular expressions to filter out the required parameters. 

```{.python .input  n=8}
print(net.collect_params('.*weight'))
print(net.collect_params('dense0.*'))
```

```{.json .output n=8}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "sequential0_ (\n  Parameter dense0_weight (shape=(256, 20), dtype=float32)\n  Parameter dense1_weight (shape=(10, 256), dtype=float32)\n)\nsequential0_ (\n  Parameter dense0_weight (shape=(256, 20), dtype=float32)\n  Parameter dense0_bias (shape=(256,), dtype=float32)\n)\n"
 }
]
```

## Parameter Initialization

Now that we know how to access the parameters, let's look at how to initialize them properly. We discussed the need for [Initialization](../chapter_deep-learning-basics/numerical-stability-and-init.md) in the previous chapter. By default, MXNet initializes the weight matrices uniformly by drawing from $U[-0.07, 0.07]$ and the bias parameters are all set to $0$. However, we often need to use other methods to initialize the weights. MXNet's `init` module provides a variety of preset initialization methods, but if we want something out of the ordinary, we need a bit of extra work. 

### Built-in Initialization

Let's begin with the built-in initializers. The code below initializes all parameters with Gaussian random variables. 

```{.python .input  n=9}
# force_reinit ensures that the variables are initialized again, regardless of whether they were 
# already initialized previously.
net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
net[0].weight.data()[0]
```

```{.json .output n=9}
[
 {
  "data": {
   "text/plain": "\n[ 0.00195949 -0.0173764   0.00047347  0.00145809  0.00326049  0.00457878\n -0.00894258  0.00493839 -0.00904343 -0.01214079  0.02156406  0.01093822\n  0.01827143 -0.0104467   0.01006219  0.0051742  -0.00806932  0.01376901\n  0.00205885  0.00994352]\n<NDArray 20 @cpu(0)>"
  },
  "execution_count": 9,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

If we wanted to initialize all parameters to 1, we could do this simply by changing the initializer to `Constant(1)`. 

```{.python .input  n=10}
net.initialize(init=init.Constant(1), force_reinit=True)
net[0].weight.data()[0]
```

```{.json .output n=10}
[
 {
  "data": {
   "text/plain": "\n[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]\n<NDArray 20 @cpu(0)>"
  },
  "execution_count": 10,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

If we want to initialize only a specific parameter in a different manner, we can simply set the initializer only for the appropriate subblock (or parameter) for that matter. For instance, below we initialize the second layer to a constant value of 42 and we use the `Xavier` initializer for the weights of the first layer. 

```{.python .input  n=11}
net[1].initialize(init=init.Constant(42), force_reinit=True)
net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
print(net[1].weight.data()[0,0])
print(net[0].weight.data()[0])
```

```{.json .output n=11}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[42.]\n<NDArray 1 @cpu(0)>\n\n[ 0.00512482 -0.06579044 -0.10849719 -0.09586414  0.06394844  0.06029618\n -0.03065033 -0.01086642  0.01929168  0.1003869  -0.09339568 -0.08703034\n -0.10472868 -0.09879824 -0.00352201 -0.11063069 -0.04257748  0.06548801\n  0.12987629 -0.13846186]\n<NDArray 20 @cpu(0)>\n"
 }
]
```

### Custom Initialization

Sometimes, the initialization methods we need are not provided in the `init` module. At this point, we can implement a subclass of the `Initializer` class so that we can use it like any other initialization method. Usually, we only need to implement the `_init_weight` function and modify the incoming NDArray according to the initial result. In the example below, we  pick a decidedly bizarre and nontrivial distribution, just to prove the point. We draw the coefficients from the following distribution:

$$
\begin{aligned}
    w \sim \begin{cases}
        U[5, 10] & \text{ with probability } \frac{1}{4} \\
            0    & \text{ with probability } \frac{1}{2} \\
        U[-10, -5] & \text{ with probability } \frac{1}{4}
    \end{cases}
\end{aligned}
$$

```{.python .input  n=12}
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = nd.random.uniform(low=-10, high=10, shape=data.shape)
        data *= data.abs() >= 5

net.initialize(MyInit(), force_reinit=True)
net[0].weight.data()[0]
```

```{.json .output n=12}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Init dense0_weight (256, 20)\nInit dense1_weight (10, 256)\n"
 },
 {
  "data": {
   "text/plain": "\n[-5.3659673  7.5773945  8.986376  -0.         8.827555   0.\n  5.9840508 -0.         0.         0.         7.4857597 -0.\n -0.         6.8910007  6.9788704 -6.1131554  0.         5.4665203\n -9.735263   9.485172 ]\n<NDArray 20 @cpu(0)>"
  },
  "execution_count": 12,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

If even this functionality is insufficient, we can set parameters directly. Since `data()` returns an NDArray we can access it just like any other matrix. A note for advanced users - if you want to adjust parameters within an `autograd` scope you need to use `set_data` to avoid confusing the automatic differentiation mechanics. 

```{.python .input  n=17}
net[0].weight.data()[:] += 1
net[0].weight.data()[0,0] = 42
net[0].weight.data()[0]
```

```{.json .output n=17}
[
 {
  "data": {
   "text/plain": "\n[42.         2.950582   3.0638967  2.9523818  2.9561982  2.9481177\n  3.0565577  3.0110493  3.0061328  2.9429164  2.9939675  3.0574763\n  3.0534859  2.941616   2.9942045  3.042053   3.0313835  3.0527983\n  2.9858637  2.9430842]\n<NDArray 20 @cpu(0)>"
  },
  "execution_count": 17,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## Tied Parameters

In some cases, we want to share model parameters across multiple layers. For instance when we want to find good word embeddings we may decide to use the same parameters both for encoding and decoding of words. We discussed one such case when we introduced [Blocks](model-construction.md). Let's see how to do this a bit more elegantly. In the following we allocate a dense layer and then use its parameters specifically to set those of another layer. 

```{.python .input  n=19}
net = nn.Sequential()
# we need to give the shared layer a name such that we can reference its parameters
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()

x = nd.random.uniform(shape=(2, 20))
net(x)

# Check whether the parameters are the same
print(net[1].weight.data()[0] == net[2].weight.data()[0])
net[1].weight.data()[0,0] = 100
# And make sure that they're actually the same object rather than just having the same value.
print(net[1].weight.data()[0] == net[2].weight.data()[0])
```

```{.json .output n=19}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[1. 1. 1. 1. 1. 1. 1. 1.]\n<NDArray 8 @cpu(0)>\n\n[1. 1. 1. 1. 1. 1. 1. 1.]\n<NDArray 8 @cpu(0)>\n"
 }
]
```

The above exampe shows that the parameters of the second and third layer are tied. They are identical rather than just being equal. That is, by changing one of the parameters the other one changes, too. What happens to the gradients is quite ingenious. Since the model parameters contain gradients, the gradients of the second hidden layer and the third hidden layer are accumulated in the `shared.params.grad( )` during backpropagation.

## Parameter Serialization




## Summary

* We have several ways to access, initialize, and share model parameters.
* We can customize the initialization method.


## exercise

* Refer to the MXNet documentation regarding the `init` module for different parameter initialization methods.
* Try accessing the model parameters after`net.initialize()` and before `net(x)` to observe the shape of the model parameters.
* Construct a multilayer perceptron containing a shared parameter layer and train it. During the training process, observe the model parameters and gradients of each layer.

## Scan the QR Code to Access [Discussions](https://discuss.gluon.ai/t/topic/987)

![](../img/qr_parameters.svg)
