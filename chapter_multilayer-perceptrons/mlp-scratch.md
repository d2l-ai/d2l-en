# Implementation of Multilayer Perceptron from Scratch
:label:`sec_mlp_scratch`

Now that we have characterized 
multilayer perceptrons (MLPs) mathematically, 
let us try to implement one ourselves.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

To compare against our previous results
achieved with (linear) softmax regression
(:numref:`sec_softmax_scratch`),
we will continue work with 
the Fashion-MNIST image classification dataset 
(:numref:`sec_fashion_mnist`).

```{.python .input}
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

```{.python .input}
#@tab pytorch
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

```{.python .input}
#@tab tensorflow
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## Initializing Model Parameters

Recall that Fashion-MNIST contains $10$ classes,
and that each image consists of a $28 \times 28 = 784$
grid of (black and white) pixel values.
Again, we will disregard the spatial structure
among the pixels (for now),
so we can think of this as simply a classification dataset
with $784$ input features and $10$ classes.
To begin, we will implement an MLP
with one hidden layer and $256$ hidden units.
Note that we can regard both of these quantities
as *hyperparameters* and ought in general
to set them based on performance on validation data.
Typically, we choose layer widths in powers of $2$,
which tend to be computationally efficient because
of how memory is alotted and addressed in hardware.

Again, we will represent our parameters with several `ndarray`s.
Note that *for every layer*, we must keep track of
one weight matrix and one bias vector.
As always, we allocate memory
for the gradients (of the loss) with respect to these parameters.

```{.python .input}
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = np.random.normal(scale=0.01, size=(num_inputs, num_hiddens))
b1 = np.zeros(num_hiddens)
W2 = np.random.normal(scale=0.01, size=(num_hiddens, num_outputs))
b2 = np.zeros(num_outputs)
params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()
```

```{.python .input}
#@tab pytorch
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True)*0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True)*0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]
```

```{.python .input}
#@tab tensorflow
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = tf.Variable(tf.random.normal(shape=(num_inputs, num_hiddens), mean=0, stddev=.01, dtype=tf.float32))
b1 = tf.Variable(tf.zeros(num_hiddens, dtype=tf.float32))
W2 = tf.Variable(tf.random.normal(shape=(num_hiddens, num_outputs), mean=0, stddev=.01, dtype=tf.float32))
b2 = tf.Variable(tf.random.normal([num_outputs], stddev=.1))

params = [W1, b1, W2, b2]
```

## Activation Function

To make sure we know how everything works,
we will implement the ReLU activation ourselves
using the maximum function rather than 
invoking `relu` directly.

```{.python .input}
def relu(X):
    return np.maximum(X, 0)
```

```{.python .input}
#@tab pytorch
def relu(X):
    a=torch.zeros_like(X)
    return torch.max(X, a)
```

```{.python .input}
#@tab tensorflow
def relu(X):
    return tf.math.maximum(X, 0)
```

## The model

Because we are disregarding spatial structure, 
we `reshape` each 2D image into 
a flat vector of length  `num_inputs`.
Finally, we implement our model 
with just a few lines of code.

```{.python .input}
def net(X):
    X = X.reshape(-1, num_inputs)
    H = relu(np.dot(X, W1) + b1)
    return np.dot(H, W2) + b2
```

```{.python .input}
#@tab pytorch
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X@W1 + b1)   # Here '@' stands for dot product operation
    return (H@W2 + b2)
```

```{.python .input}
#@tab tensorflow
def net(X):
    X = tf.reshape(X, shape=[-1, num_inputs])
    H = relu(tf.matmul(tf.cast(X, dtype=tf.float32), W1) + b1)
    return tf.math.softmax(tf.matmul(H, W2) + b2)
```

## The Loss Function

To ensure numerical stability,
and because we already implemented
the softmax function from scratch
(:numref:`sec_softmax_scratch`),
we leverage Gluon's integrated function
for calculating the softmax and cross-entropy loss.
Recall our earlier discussion of these intricacies 
(:numref:`sec_mlp`).
We encourage the interested reader 
to examine the source code for loss function
to deepen their knowledge of implementation details.

```{.python .input}
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

```{.python .input}
#@tab pytorch
loss = nn.CrossEntropyLoss()
```

```{.python .input}
#@tab tensorflow
def loss(y_hat, y):
    return tf.losses.sparse_categorical_crossentropy(y, y_hat)
```

## Training

Fortunately, the training loop for MLPs
is exactly the same as for softmax regression.
Leveraging the `d2l` package again, 
we call the `train_ch3` function  
(see :numref:`sec_softmax_scratch`),
setting the number of epochs to $10$ 
and the learning rate to $0.5$.

```{.python .input}
num_epochs, lr = 10, 0.5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs,
              lambda batch_size: d2l.sgd(params, lr, batch_size))
```

```{.python .input}
#@tab pytorch
num_epochs, lr = 10, 0.5
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
```

```{.python .input}
#@tab tensorflow
num_epochs, lr = 10, 0.5
updater = tf.keras.optimizers.SGD(learning_rate=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater, params=[W1, W2, b1, b2])
```

To evaluate the learned model, 
we apply it on some test data.

```{.python .input}
d2l.predict_ch3(net, test_iter)
```

```{.python .input}
#@tab pytorch
d2l.predict_ch3(net, test_iter)
```

```{.python .input}
#@tab tensorflow
d2l.predict_ch3(net, test_iter)
```

This looks a bit better than our previous result,
which used simple linear models, and it gives us 
some signal that we are on the right path.

## Summary

We saw that implementing a simple MLP is easy, 
even when done manually.
That said, with a large number of layers, 
this can still get messy 
(e.g., naming and keeping track of our model's parameters, etc).

## Exercises

1. Change the value of the hyperparameter `num_hiddens` and see how this hyperparameter influences your results. Determine the best value of this hyperparameter, keeping all others constant.
1. Try adding an additional hidden layer to see how it affects the results.
1. How does changing the learning rate alter your results? Fixing the model architecture and other hyperparameters (including number of epochs), what learning rate gives you the best results? 
1. What is the best result you can get by optimizing over all the parameters (learning rate, iterations, number of hidden layers, number of hidden units per layer) jointly? 
1. Describe why it is much more challenging to deal with multiple hyperparameters. 
1. What is the smartest strategy you can think of for structuring a search over multiple hyperparameters?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/92)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/93)
:end_tab:
