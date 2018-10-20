# Gluon Implementation of Multilayer Perceptron

In the following section, we use Gluon to implement a multilayer perceptron in the previous section.  First, we import the required packages or modules.

```{.python .input}
import gluonbook as gb
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn
```

## Define the model

The only difference from the softmax regression is the addition of a fully connected layer as a hidden layer.  It has 256 hidden units and uses ReLU as the activation function.

```{.python .input  n=5}
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'),
        nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

## Reading Data and Training Model

We use almost the same steps for softmax regression training as we do for reading and training the model.

```{.python .input  n=6}
batch_size = 256
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)

loss = gloss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
num_epochs = 5
gb.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
             None, None, trainer)
```

## Summary

* It is easier to construct the multilayer perceptron using Gluon. 

## exercise

* Try adding a few more hidden layers to compare the implementation that started from scratch in the previous section.  
* Use other activation functions to see resulting effects. 

## Scan the QR Code to Access [Discussions](https://discuss.gluon.ai/t/topic/738)

![](../img/qr_mlp-gluon.svg)
