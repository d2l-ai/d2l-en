# Concise Implementation of Multilayer Perceptron
:label:`sec_mlp_gluon`

As you might expect, by relying on the Gluon library,
we can implement MLPs even more concisely.

```{.python .input}
import d2l
from mxnet import gluon, init, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
import d2l_pytorch as d2l
import torch
import torch.nn as nn
```

## The Model

As compared to our gluon implementation 
of softmax regression implementation
(:numref:`sec_softmax_gluon`),
the only difference is that we add 
*two* `Dense` (fully-connected) layers 
(previously, we added *one*).
The first is our hidden layer, 
which contains *256* hidden units
and applies the ReLU activation function.
The second, is our output layer.

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'),
        nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

```{.python .input}
#@tab pytorch
class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1,784)
    
net = nn.Sequential(Reshape(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)
```

Note that Gluon, as usual, automatically
infers the missing input dimensions to each layer.

The training loop is *exactly* the same
as when we implemented softmax regression.
This modularity enables us to separate 
matters concerning the model architecture
from orthogonal considerations.

```{.python .input}
batch_size, num_epochs = 256, 10
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

```{.python .input}
#@tab pytorch
num_epochs, lr, batch_size = 10, 0.5, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

## Exercises

1. Try adding different numbers of hidden layers. What setting (keeping other parameters and hyperparameters constant) works best? 
1. Try out different activation functions. Which ones work best?
1. Try different schemes for initializing the weights. What method works best?

## [Discussions](https://discuss.mxnet.io/t/2340)

![](../img/qr_mlp-gluon.svg)
