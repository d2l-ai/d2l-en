# Concise Implementation of Multilayer Perceptron

Now that we learned how multilayer perceptrons (MLPs) work in theory, let's implement them. We begin, as always, by importing modules.

```{.python .input}
import sys
sys.path.insert(0, '..')

import d2l
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn
```

## The Model

The only difference from the softmax regression is the addition of a fully connected layer as a hidden layer.  It has 256 hidden units and uses ReLU as the activation function.

```{.python .input  n=5}
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

One minor detail is of note when invoking `net.add()`. It adds one or more layers to the network. That is, an equivalent to the above lines would be `net.add(nn.Dense(256, activation='relu'), nn.Dense(10))`. Also note that Gluon automagically infers the missing parameteters, such as the fact that the second layer needs a matrix of size $256 \times 10$. This happens the first time the network is invoked.

We use almost the same steps for softmax regression training as we do for reading and training the model.

```{.python .input  n=6}
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

loss = gloss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None,
              None, trainer)
```

## Exercises

1. Try adding a few more hidden layers to see how the result changes.
1. Try out different activation functions. Which ones work best?
1. Try out different initializations of the weights.

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2340)

![](../img/qr_mlp-gluon.svg)
