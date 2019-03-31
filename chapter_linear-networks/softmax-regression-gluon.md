# Concise Implementation of Softmax Regression

We already saw that it is much more convenient to use Gluon in the context of [linear regression](linear-regression-gluon.md). Now we will see how this applies to classification, too. We begin with our import ritual.

```{.python .input  n=1}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import d2l
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn
```

We still use the Fashion-MNIST data set and the batch size set from the last section.

```{.python .input  n=2}
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## Initialize Model Parameters

As [mentioned previously](softmax-regression.md), the output layer of softmax regression is a fully connected layer. Therefore, we are adding a fully connected layer with 10 outputs. We initialize the weights at random with zero mean and standard deviation 0.01.

```{.python .input  n=3}
net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

## The Softmax

In the previous example, we calculated our model's output and then ran this output through the cross-entropy loss. At its heart it uses `-nd.pick(y_hat, y).log()`. Mathematically, that's a perfectly reasonable thing to do. However, computationally, things can get hairy, as we've already alluded to a few times (e.g. in the context of [Naive Bayes](../chapter_crashcourse/naive-bayes.md) and in the problem set of the previous chapter). Recall that the softmax function calculates $\hat y_j = \frac{e^{z_j}}{\sum_{i=1}^{n} e^{z_i}}$, where $\hat y_j$ is the j-th element of ``yhat`` and $z_j$ is the j-th element of the input ``y_linear`` variable, as computed by the softmax.

If some of the $z_i$ are very large (i.e. very positive), $e^{z_i}$ might be larger than the largest number we can have for certain types of ``float`` (i.e. overflow). This would make the denominator (and/or numerator) ``inf`` and we get zero, or ``inf``, or ``nan`` for $\hat y_j$. In any case, we won't get a well-defined return value for ``cross_entropy``. This is the reason we subtract $\text{max}(z_i)$ from all $z_i$ first in ``softmax`` function. You can verify that this shifting in $z_i$ will not change the return value of ``softmax``.

After the above subtraction/ normalization step, it is possible that $z_j$ is very negative. Thus, $e^{z_j}$ will be very close to zero and might be rounded to zero due to finite precision (i.e underflow), which makes $\hat y_j$ zero and we get ``-inf`` for $\text{log}(\hat y_j)$. A few steps down the road in backpropagation, we start to get horrific not-a-number (``nan``) results printed to screen.

Our salvation is that even though we're computing these exponential functions, we ultimately plan to take their log in the cross-entropy functions. It turns out that by combining these two operators ``softmax`` and ``cross_entropy`` together, we can elude the numerical stability issues that might otherwise plague us during backpropagation. As shown in the equation below, we avoided calculating $e^{z_j}$ but directly used $z_j$ due to $\log(\exp(\cdot))$.

$$
\begin{aligned}
\log{(\hat y_j)} & = \log\left( \frac{e^{z_j}}{\sum_{i=1}^{n} e^{z_i}}\right) \\
& = \log{(e^{z_j})}-\text{log}{\left( \sum_{i=1}^{n} e^{z_i} \right)} \\
& = z_j -\log{\left( \sum_{i=1}^{n} e^{z_i} \right)}
\end{aligned}
$$

We'll want to keep the conventional softmax function handy in case we ever want to evaluate the probabilities output by our model. But instead of passing softmax probabilities into our new loss function, we'll just pass $\hat{y}$ and compute the softmax and its log all at once inside the softmax_cross_entropy loss function, which does smart things like the log-sum-exp trick ([see on Wikipedia](https://en.wikipedia.org/wiki/LogSumExp)).

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
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None,
              None, trainer)
```

Just as before, this algorithm converges to a fairly decent accuracy of 83.7%, albeit this time with a lot fewer lines of code than before. Note that in many cases Gluon takes specific precautions beyond what one would naively do to ensure numerical stability. This takes care of many common pitfalls when coding a model from scratch.


## Exercises

1. Try adjusting the hyper-parameters, such as batch size, epoch, and learning rate, to see what the results are.
1. Why might the test accuracy decrease again after a while? How could we fix this?

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2337)

![](../img/qr_softmax-regression-gluon.svg)
