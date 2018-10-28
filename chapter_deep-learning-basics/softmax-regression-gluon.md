# Softmax Regression in Gluon

We already saw that it is much more convenient to use Gluon in the context of [linear regression](linear-regression-gluon.md). Now we will see how this applies to classification, too. We being with our import ritual.

```{.python .input  n=1}
%matplotlib inline
import gluonbook as gb
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn
```

We still use the Fashion-MNIST data set and the batch size set from the last section.

```{.python .input  n=2}
batch_size = 256
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)
```

## Initialize Model Parameters

As [mentioned previously](softmax-regression.md), the output layer of softmax regression is a fully connected layer. Therefore, we are adding a fully connected layer with 10 outputs. We initialize the weights at random with zero mean and standard deviation 0.01.

```{.python .input  n=3}
net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

## The Softmax

As the problem set from the last section illustrates, computing softmax and loss function separately can cause all sorts of numerical instabilities, mostly due to numerical overflow (e.g. $\exp(50)$) and underflow (e.g. $\exp(-50)$). To address this Gluon provides a function that includes softmax operations and cross-entropy loss calculations. This results in better numerical stability and better computational efficiency.

```{.python .input  n=4}
loss = gloss.SoftmaxCrossEntropyLoss()
```

## Optimization Algorithm

We use the mini-batch random gradient descent with a learning rate of 0.1 as the optimization algorithm. Note that this is the same choice as for linear regression and it illustrates the portability of the optimizers.

```{.python .input  n=5}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
```

## Training

Next, we use the training functions defined in the last section to train a model.

```{.python .input  n=6}
num_epochs = 5
gb.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None,
             None, trainer)
```

Just as before, this algorithm converges to a fairly decent accuracy of 83.7%, albeit this time with a lot fewer lines of code than before. Note that in many cases Gluon takes specific precautions beyond what one would naively do to ensure numerical stability. This takes care of many common pitfalls when coding a model from scratch. 


## Problems

1. Try adjusting the hyper-parameters, such as batch size, epoch, and learning rate, to see what the results are.
1. Why might the test accuracy decrease again after a while? How could we fix this? 

## Scan the QR Code to Access [Discussions](https://discuss.gluon.ai/t/topic/740)

![](../img/qr_softmax-regression-gluon.svg)

```{.python .input}

```
