# Gluon Implementation of Softmax Regression

We have introduced the convenience of using Gluon to implement models in the ["Gluon Implementation of Linear Regression"](linear-regression-gluon.md) section. Here, we will again use Gluon to implement a softmax regression model. First, we must import the packages or modules required for the implementation in this section.

```{.python .input  n=1}
%matplotlib inline
import gluonbook as gb
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn
```

## Acquire and Read the Data

We still use the Fashion-MNIST data set and the batch size set from the last section.

```{.python .input  n=2}
batch_size = 256
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)
```

## Define and Initialize the Model

We have mentioned before that the output layer of softmax regression is a fully connected layer in the ["Softmax regression"](softmax-regression.md) section. Therefore, we are adding a fully connected layer with 10 outputs. We use the weight parameter from the random initialization and normal distribution model with a mean of 0 and a standard deviation of 0.01.

```{.python .input  n=3}
net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

## Softmax and cross-entropy loss function

If you have done the exercise from the last section, you probably noticed that separate definitions for the softmax operation and cross-entropy loss function may cause numerical instability. Therefore, Gluon has provided a function that includes softmax operations and cross-entropy loss calculations. This results in better numerical stability.

```{.python .input  n=4}
loss = gloss.SoftmaxCrossEntropyLoss()
```

## Define the Optimization the Algorithm

We use the mini-batch random gradient descent with a learning rate of 0.1 as the optimization algorithm.

```{.python .input  n=5}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
```

## To train a model

Next, we use the training functions defined in the last section to train a model.

```{.python .input  n=6}
num_epochs = 5
gb.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None,
             None, trainer)
```

## Summary

* The function provided by Gluon tends to have better numerical stability.
* We could use Gluon to implement softmax regression more succinctly. 

## exercise

* Try adjusting the hyper-parameters, such as batch size, epoch, and learning rate, to see what the results are.

## Scan the QR Code to Access [Discussions](https://discuss.gluon.ai/t/topic/740)

![](../img/qr_softmax-regression-gluon.svg)
