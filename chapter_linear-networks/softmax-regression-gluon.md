# Concise Implementation of Softmax Regression

Just as Gluon made it much easier
to implement linear regression in :numref:`sec_linear_gluon`,
we will find it similarly (or possibly more)
convenient for implementing classification models.
Again, we begin with our import ritual.

```{.python .input  n=1}
import d2l
from mxnet import gluon, init, npx
from mxnet.gluon import nn
npx.set_np()
```

Let us stick with the Fashion-MNIST dataset 
and keep the batch size at $256$ as in the last section.

```{.python .input  n=2}
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## Initialize Model Parameters

As mentioned in :numref:`sec_softmax`,
the output layer of softmax regression 
is a fully connected (`Dense`) layer.
Therefore, to implement our model,
we just need to add one `Dense` layer 
with 10 outputs to our `Sequential`.
Again, here, the `Sequential` is not really necessary,
but we might as well form the habit since it will be ubiquitous
when implementing deep models.
Again, we initialize the weights at random
with zero mean and standard deviation 0.01.

```{.python .input  n=3}
net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

## The Softmax

In the previous example, we calculated our model's output
and then ran this output through the cross-entropy loss.
At its heart it uses `-nd.pick(y_hat, y).log()`.
Mathematically, that is a perfectly reasonable thing to do.
However, computationally, things can get hairy
when dealing with exponentiation
due to numerical stability issues,
as discussed  in :numref:`sec_naive_bayes`) and in the exercises.
Recall that the softmax function calculates
$\hat y_j = \frac{e^{z_j}}{\sum_{i=1}^{n} e^{z_i}}$, 
where $\hat y_j$ is the $j^\mathrm{th}$ element of ``yhat`` 
and $z_j$ is the $j^\mathrm{th}$ element of the input
``y_linear`` variable, as computed by the softmax.

If some of the $z_i$ are very large (i.e., very positive),
$e^{z_i}$ might be larger than the largest number
we can have for certain types of ``float`` (i.e., overflow).
This would make the denominator (and/or numerator) ``inf`` and we get zero,
or ``inf``, or ``nan`` for $\hat y_j$.
In any case, we will not get a well-defined return value for ``cross_entropy``.
This is the reason we subtract $\text{max}(z_i)$
from all $z_i$ first in the ``softmax`` function.
You can verify that this shifting in $z_i$
will not change the return value of ``softmax``.

After these subtraction and normalization step,
it might be that possible that some $z_j$ have laege negative values
and thus that the corresponding $e^{z_j}$ will take values close to zero.
These might be rounded to zero due to finite precision (i.e underflow),
making $\hat y_j$ zero and giving us ``-inf`` for $\text{log}(\hat y_j)$.
A few steps down the road in backpropagation,
we start to get horrific not-a-number (``nan``) results printed to screen.

What saves us here is that even though we are computing exponential functions, 
we ultimately plan to take their log when we calculate the cross-entropy loss.
It turns out that by combining these two operators
``softmax`` and ``cross_entropy`` together,
we can escape the numerical stability issues
that might otherwise plague us during backpropagation.
As shown in the equation below, we avoided calculating $e^{z_j}$
but directly used $z_j$ due to $\log(\exp(\cdot))$.

$$
\begin{aligned}
\log{(\hat y_j)} & = \log\left( \frac{e^{z_j}}{\sum_{i=1}^{n} e^{z_i}}\right) \\
& = \log{(e^{z_j})}-\text{log}{\left( \sum_{i=1}^{n} e^{z_i} \right)} \\
& = z_j -\log{\left( \sum_{i=1}^{n} e^{z_i} \right)}
\end{aligned}
$$

We will want to keep the conventional softmax function handy
in case we ever want to evaluate the probabilities output by our model.
But instead of passing softmax probabilities into our new loss function,
we will just pass $\hat{y}$ and compute the softmax and its log
all at once inside the softmax_cross_entropy loss function,
which does smart things like the log-sum-exp trick ([see on Wikipedia](https://en.wikipedia.org/wiki/LogSumExp)).

```{.python .input  n=4}
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

## Optimization Algorithm

Here, we use minibatch stochastic gradient descent
with a learning rate of $0.1$ as the optimization algorithm.
Note that this is the same as we applied in the linear regression example
and it illustrates the general applicability of the optimizers.

```{.python .input  n=5}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
```

## Training

Next we call the training function defined in the last section to train a model.

```{.python .input  n=6}
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

As before, this algorithm converges to a solution
that achieves an accuracy of 83.7%,
albeit this time with a lot fewer lines of code than before.
Note that in many cases, Gluon takes specific precautions
in addition to the most well-known tricks for ensuring numerical stability.
This saves us from many common pitfalls that might befall us
if we were to code all of our models from scratch.

## Exercises

1. Try adjusting the hyper-parameters, such as batch size, epoch, and learning rate, to see what the results are.
1. Why might the test accuracy decrease again after a while? How could we fix this?

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2337)

![](../img/qr_softmax-regression-gluon.svg)
