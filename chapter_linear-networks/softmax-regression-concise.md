# Concise Implementation of Softmax Regression
:label:`sec_softmax_gluon`

Just as Gluon made it much easier
to implement linear regression in :numref:`sec_linear_gluon`,
we will find it similarly (or possibly more)
convenient for implementing classification models.
Again, we begin with our import ritual.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, init, npx
from mxnet.gluon import nn
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

Let us stick with the Fashion-MNIST dataset
and keep the batch size at $256$ as in the last section.

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

As mentioned in :numref:`sec_softmax`,
the output layer of softmax regression
is a fully-connected layer.
Therefore, to implement our model,
we just need to add one fully-connected layer
with 10 outputs to our `Sequential`.
Again, here, the `Sequential` is not really necessary,
but we might as well form the habit since it will be ubiquitous
when implementing deep models.
Again, we initialize the weights at random
with zero mean and standard deviation $0.01$.

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

```{.python .input}
#@tab pytorch
# PyTorch doesn't implicitly reshape the inputs.
# Thus we define a layer to reshape the inputs in our network.
class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1,784)

net = nn.Sequential(Reshape(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential()
net.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
weight_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
net.add(tf.keras.layers.Dense(10, kernel_initializer= weight_initializer))
```

## The Softmax

In the previous example, we calculated our model's output
and then ran this output through the cross-entropy loss.
Mathematically, that is a perfectly reasonable thing to do.
However, from a computational perspective,
exponentiation can be a source of numerical stability issues
(as discussed  in :numref:`sec_naive_bayes`).
Recall that the softmax function calculates
$\hat y_j = \frac{e^{z_j}}{\sum_{i=1}^{n} e^{z_i}}$,
where $\hat y_j$ is the $j^\mathrm{th}$ element of ``y_hat``
and $z_j$ is the $j^\mathrm{th}$ element of the input
``y_linear`` variable, as computed by the softmax.

If some of the $z_i$ are very large (i.e., very positive),
then $e^{z_i}$ might be larger than the largest number
we can have for certain types of ``float`` (i.e., overflow).
This would make the denominator (and/or numerator) ``inf``
and we wind up encountering either $0$, ``inf``, or ``nan`` for $\hat y_j$.
In these situations we do not get a well-defined
return value for ``cross_entropy``.
One trick to get around this is to first subtract $\text{max}(z_i)$
from all $z_i$ before proceeding with the ``softmax`` calculation.
You can verify that this shifting of each $z_i$ by constant factor
does not change the return value of ``softmax``.

After the subtraction and normalization step,
it might be that possible that some $z_j$ have large negative values
and thus that the corresponding $e^{z_j}$ will take values close to zero.
These might be rounded to zero due to finite precision (i.e underflow),
making $\hat y_j$ zero and giving us ``-inf`` for $\text{log}(\hat y_j)$.
A few steps down the road in backpropagation,
we might find ourselves faced with a screenful
of the dreaded not-a-number (``nan``) results.

Fortunately, we are saved by the fact that
even though we are computing exponential functions,
we ultimately intend to take their log
(when calculating the cross-entropy loss).
By combining these two operators
(``softmax`` and ``cross_entropy``) together,
we can escape the numerical stability issues
that might otherwise plague us during backpropagation.
As shown in the equation below, we avoided calculating $e^{z_j}$
and can instead $z_j$ directly due to the canceling in $\log(\exp(\cdot))$.

$$
\begin{aligned}
\log{(\hat y_j)} & = \log\left( \frac{e^{z_j}}{\sum_{i=1}^{n} e^{z_i}}\right) \\
& = \log{(e^{z_j})}-\text{log}{\left( \sum_{i=1}^{n} e^{z_i} \right)} \\
& = z_j -\log{\left( \sum_{i=1}^{n} e^{z_i} \right)}.
\end{aligned}
$$

We will want to keep the conventional softmax function handy
in case we ever want to evaluate the probabilities output by our model.
But instead of passing softmax probabilities into our new loss function,
we will just pass the logits and compute the softmax and its log
all at once inside the softmax_cross_entropy loss function,
which does smart things like the log-sum-exp trick ([see on Wikipedia](https://en.wikipedia.org/wiki/LogSumExp)).

```{.python .input}
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

```{.python .input}
#@tab pytorch
loss = nn.CrossEntropyLoss()
```

```{.python .input}
#@tab tensorflow
net.add(tf.keras.layers.Softmax())
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```

## Optimization Algorithm

Here, we use minibatch stochastic gradient descent
with a learning rate of $0.1$ as the optimization algorithm.
Note that this is the same as we applied in the linear regression example
and it illustrates the general applicability of the optimizers.

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD(learning_rate=.1)
```

## Training

Next we call the training function defined in the last section to train a model.

```{.python .input}
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

```{.python .input}
#@tab pytorch
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

```{.python .input}
#@tab tensorflow
net.compile(optimizer=trainer, loss=loss, metrics=['accuracy'])
num_epochs = 10
net.fit(train_iter, epochs=num_epochs)
```

As before, this algorithm converges to a solution
that achieves an accuracy of 83.7%,
albeit this time with fewer lines of code than before.
Note that in many cases, Gluon takes additional precautions
beyond these most well-known tricks to ensure numerical stability,
saving us from even more pitfalls that we would encounter
if we tried to code all of our models from scratch in practice.

## Exercises

1. Try adjusting the hyper-parameters, such as batch size, epoch, and learning rate, to see what the results are.
1. Why might the test accuracy decrease again after a while? How could we fix this?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/52)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/53)
:end_tab:
