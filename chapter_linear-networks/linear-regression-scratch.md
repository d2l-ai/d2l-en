# Linear Regression Implementation from Scratch
:label:`sec_linear_scratch`

Now that you understand the key ideas behind linear regression,
we can begin to work through a hands-on implementation in code.
In this section, we will implement the entire method from scratch,
including the data pipeline, the model,
the loss function, and the gradient descent optimizer.
While modern deep learning frameworks can automate nearly all of this work,
implementing things from scratch is the only
to make sure that you really know what you are doing.
Moreover, when it comes time to customize models,
defining our own layers, loss functions, etc.,
understanding how things work under the hood will prove handy.
In this section, we will rely only on `ndarray` and auto differentiation.
Afterwards, we will introduce a more compact implementation,
taking advantage of framework's bells and whistles.
To start off, we import the few required packages.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
import random
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import random
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

## Generating the Dataset

To keep things simple, we will construct an artificial dataset
according to a linear model with additive noise.
Out task will be to recover this model's parameters
using the finite set of examples contained in our dataset.
We will keep the data low-dimensional so we can visualize it easily.
In the following code snippet, we generated a dataset
containing $1000$ examples, each consisting of $2$ features
sampled from a standard normal distribution.
Thus our synthetic dataset will be an object
$\mathbf{X}\in \mathbb{R}^{1000 \times 2}$.

The true parameters generating our data will be
$\mathbf{w} = [2, -3.4]^\top$ and $b = 4.2$
and our synthetic labels will be assigned according
to the following linear model with noise term $\epsilon$:

$$\mathbf{y}= \mathbf{X} \mathbf{w} + b + \mathbf\epsilon.$$

You could think of $\epsilon$ as capturing potential
measurement errors on the features and labels.
We will assume that the standard assumptions hold and thus
that $\epsilon$ obeys a normal distribution with mean of $0$.
To make our problem easy, we will set its standard deviation to $0.01$.
The following code generates our synthetic dataset:

```{.python .input}
def synthetic_data(w, b, num_examples):  #@save
    """Generate y = X w + b + noise."""
    X = np.random.normal(0, 1, (num_examples, len(w)))
    y = np.dot(X, w) + b
    y += np.random.normal(0, 0.01, y.shape)
    return X, y

true_w = np.array([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
```

```{.python .input}
#@tab pytorch
def synthetic_data(w, b, num_examples):  #@save
    """Generate y = X w + b + noise."""
    X = torch.zeros(size=(num_examples, len(w))).normal_()
    y = torch.matmul(X, w) + b
    y += torch.zeros(size=y.shape).normal_(std=0.01)
    return X, y

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
```

```{.python .input}
#@tab tensorflow
def synthetic_data(w, b, num_examples):  #@save
    """Generate y = X w + b + noise."""
    X = tf.zeros(shape=(num_examples, w.shape[0]))
    X += tf.random.normal(shape=X.shape)
    y = tf.matmul(X, w) + b
    y += tf.random.normal(shape=y.shape, stddev=0.01)
    y = tf.reshape(y, [num_examples])
    return X, y

true_w = tf.constant([2, -3.4], shape=(2, 1))
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
```

Note that each row in `features` consists of a 2-dimensional data point
and that each row in `labels` consists of a 1-dimensional target value (a scalar).

```{.python .input}
print('features:', features[0],'\nlabel:', labels[0])
```

```{.python .input}
#@tab pytorch
print('features:', features[0],'\nlabel:', labels[0])
```

```{.python .input}
#@tab tensorflow
print('features:', features[0],'\nlabel:', labels[0])
```

By generating a scatter plot using the second feature `features[:, 1]` and `labels`,
we can clearly observe the linear correlation between the two.

```{.python .input}
d2l.set_figsize((3.5, 2.5))
d2l.plt.scatter(features[:, 1].asnumpy(), labels.asnumpy(), 1);
```

```{.python .input}
#@tab pytorch
d2l.set_figsize((3.5, 2.5))
d2l.plt.scatter(features[:, 1].numpy(), labels.numpy(), 1);
```

```{.python .input}
#@tab tensorflow
d2l.set_figsize((3.5, 2.5))
d2l.plt.scatter(features[:, 1].numpy(), labels.numpy(), 1);
```

## Reading the Dataset

Recall that training models consists of
making multiple passes over the dataset,
grabbing one minibatch of examples at a time,
and using them to update our model.
Since this process is so fundamental
to training machine learning algorithms,
its worth defining a utility function
to shuffle the data and access it in minibatches.

In the following code, we define a `data_iter` function
to demonstrate one possible implementation of this functionality.
The function takes a batch size, a design matrix,
and a vector of labels, yielding minibatches of size `batch_size`.
Each minibatch consists of a tuple of features and labels.

```{.python .input}
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = np.array(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
```

```{.python .input}
#@tab pytorch
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[j], labels[j]
```

```{.python .input}
#@tab tensorflow
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = tf.constant(indices[i: min(i + batch_size, num_examples)])
        yield tf.gather(features, j), tf.gather(labels, j)
```

In general, note that we want to use reasonably sized minibatches
to take advantage of the GPU hardware,
which excels at parallelizing operations.
Because each example can be fed through our models in parallel
and the gradient of the loss function for each example can also be taken in parallel,
GPUs allow us to process hundreds of examples in scarcely more time
than it might take to process just a single example.

To build some intuition, let us read and print
the first small batch of data examples.
The shape of the features in each minibatch tells us
both the minibatch size and the number of input features.
Likewise, our minibatch of labels will have a shape given by `batch_size`.

```{.python .input}
batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
```

```{.python .input}
#@tab pytorch
batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
```

```{.python .input}
#@tab tensorflow
batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
```

As we run the iterator, we obtain distinct minibatches
successively until all the data has been exhausted (try this).
While the iterator implemented above is good for didactic purposes,
it is inefficient in ways that might get us in trouble on real problems.
For example, it requires that we load all data in memory
and that we perform lots of random memory access.
The built-in iterators implemented in Apache MXNet
are considerably more efficient and they can deal
both with data stored in file and data fed via a data stream.

## Initializing Model Parameters

Before we can begin optimizing our model's parameters by gradient descent,
we need to have some parameters in the first place.
In the following code, we initialize weights by sampling
random numbers from a normal distribution with mean 0
and a standard deviation of $0.01$, setting the bias $b$ to $0$.

```{.python .input}
w = np.random.normal(0, 0.01, (2, 1))
b = np.zeros(1)
w.attach_grad()
b.attach_grad()
```

```{.python .input}
#@tab pytorch
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

```{.python .input}
#@tab tensorflow
w = tf.Variable(tf.random.normal(shape=(2, 1), mean=0, stddev=0.01), trainable=True)
b = tf.Variable(tf.zeros(1), trainable=True)
```

After initialized our parameters,
our next task is to update them until
they fit our data sufficiently well.
Each update requires taking the gradient
(a multi-dimensional derivative)
of our loss function with respect to the parameters.
Given this gradient, we can update each parameter
in the direction that reduces the loss.

Since nobody wants to compute gradients explicitly
(this is tedious and error prone),
we use automatic differentiation to compute the gradient.
See :numref:`sec_autograd` for more details.
Recall from the autograd chapter
that in order for the system to know
that it should store a gradient for our parameters, we specified to attach
gradients to both $w$ and $b$ on the above codes.

## Defining the Model

Next, we must define our model,
relating its inputs and parameters to its outputs.
Recall that to calculate the output of the linear model,
we simply take the matrix-vector dot product
of the examples $\mathbf{X}$ and the models weights $w$,
and add the offset $b$ to each example.
Note that below $Xw$  is a vector and $b$ is a scalar.
Recall that when we add a vector and a scalar,
the scalar is added to each component of the vector.

```{.python .input}
def linreg(X, w, b):  #@save
    return np.dot(X, w) + b
```

```{.python .input}
#@tab pytorch
def linreg(X, w, b):  #@save
    return torch.matmul(X, w) + b
```

```{.python .input}
#@tab tensorflow
def linreg(X, w, b):  #@save
    return tf.matmul(X, w) + b
```

## Defining the Loss Function

Since updating our model requires taking
the gradient of our loss function,
we ought to define the loss function first.
Here we will use the squared loss function
as described in the previous section.
In the implementation, we need to transform the true value `y`
into the predicted value's shape `y_hat`.
The result returned by the following function
will also be the same as the `y_hat` shape.

```{.python .input}
def squared_loss(y_hat, y):  #@save
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
```

```{.python .input}
#@tab pytorch
def squared_loss(y_hat, y):  #@save
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
```

```{.python .input}
#@tab tensorflow
def squared_loss(y_hat, y):  #@save
    return (y_hat - tf.reshape(y, y_hat.shape)) ** 2 / 2
```

## Defining the Optimization Algorithm

As we discussed in the previous section,
linear regression has a closed-form solution.
However, this is not a book about linear regression,
it is a book about deep learning.
Since none of the other models that this book introduces
can be solved analytically, we will take this opportunity to introduce your first working example of stochastic gradient descent (SGD).


At each step, using one batch randomly drawn from our dataset,
we will estimate the gradient of the loss with respect to our parameters.
Next, we will update our parameters (a small amount)
in the direction that reduces the loss.
Recall from :numref:`sec_autograd` that after we call `backward`
each parameter (`param`) will have its gradient stored in `param.grad`.
The following code applies the SGD update,
given a set of parameters, a learning rate, and a batch size.
The size of the update step is determined by the learning rate `lr`.
Because our loss is calculated as a sum over the batch of examples,
we normalize our step size by the batch size (`batch_size`),
so that the magnitude of a typical step size
does not depend heavily on our choice of the batch size.

```{.python .input}
def sgd(params, lr, batch_size):  #@save
    for param in params:
        param[:] = param - lr * param.grad / batch_size
```

```{.python .input}
#@tab pytorch
def sgd(params, lr, batch_size):  #@save
    for param in params:
        param.data.sub_(lr*param.grad/batch_size)
        param.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd(params, grads, lr, batch_size):  #@save
    for param, grad in zip(params, grads):
        param.assign_sub(lr*grad/batch_size)
```

## Training

Now that we have all of the parts in place,
we are ready to implement the main training loop.
It is crucial that you understand this code
because you will see nearly identical training loops
over and over again throughout your career in deep learning.

In each iteration, we will grab minibatches of models,
first passing them through our model to obtain a set of predictions.
After calculating the loss, we call the `backward` function
to initiate the backwards pass through the network,
storing the gradients with respect to each parameter
in its corresponding `.grad` attribute.
Finally, we will call the optimization algorithm `sgd`
to update the model parameters.
Since we previously set the batch size `batch_size` to $10$,
the loss shape `l` for each minibatch is ($10$, $1$).

In summary, we will execute the following loop:

* Initialize parameters $(\mathbf{w}, b)$
* Repeat until done
    * Compute gradient $\mathbf{g} \leftarrow \partial_{(\mathbf{w},b)} \frac{1}{\mathcal{B}} \sum_{i \in \mathcal{B}} l(\mathbf{x}^i, y^i, \mathbf{w}, b)$
    * Update parameters $(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta \mathbf{g}$

In the code below, `l` is a vector of the losses
for each example in the minibatch.
Because `l` is not a scalar variable,
running `l.backward()` adds together the elements in `l`
to obtain the new variable and then calculates the gradient.

In each epoch (a pass through the data),
we will iterate through the entire dataset
(using the `data_iter` function) once
passing through every examples in the training dataset
(assuming the number of examples is divisible by the batch size).
The number of epochs `num_epochs` and the learning rate `lr` are both hyper-parameters,
which we set here to $3$ and $0.03$, respectively.
Unfortunately, setting hyper-parameters is tricky
and requires some adjustment by trial and error.
We elide these details for now but revise them
later in
:numref:`chap_optimization`.

```{.python .input}
lr = 0.03  # Learning rate
num_epochs = 3  # Number of iterations
net = linreg  # Our fancy linear model
loss = squared_loss  # 0.5 (y-y')^2

for epoch in range(num_epochs):
    # Assuming the number of examples can be divided by the batch size, all
    # the examples in the training dataset are used once in one epoch
    # iteration. The features and tags of minibatch examples are given by X
    # and y respectively
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y)  # Minibatch loss in X and y
        l.backward()  # Compute gradient on l with respect to [w, b]
        sgd([w, b], lr, batch_size)  # Update parameters using their gradient
    train_l = loss(net(features, w, b), labels)
    print(f'epoch {epoch+1}, loss {float(train_l.mean())}')
```

```{.python .input}
#@tab pytorch
lr = 0.03  # Learning rate
num_epochs = 3  # Number of iterations
net = linreg  # Our fancy linear model
loss = squared_loss  # 0.5 (y-y')^2

for epoch in range(num_epochs):
    # Assuming the number of examples can be divided by the batch size, all
    # the examples in the training data set are used once in one epoch
    # iteration. The features and tags of mini-batch examples are given by X
    # and y respectively
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # Minibatch loss in X and y
        l.mean().backward()  # Compute gradient on l with respect to [w,b]
        sgd([w, b], lr, batch_size)  # Update parameters using their gradient
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch+1}, loss {float(train_l.mean())}')
```

```{.python .input}
#@tab tensorflow
lr = 0.03  # Learning rate
num_epochs = 3  # Number of iterations
net = linreg  # Our fancy linear model
loss = squared_loss  # 0.5 (y-y')^2
for epoch in range(num_epochs):
    # Assuming the number of examples can be divided by the batch size, all
    # the examples in the training data set are used once in one epoch
    # iteration. The features and tags of mini-batch examples are given by X
    # and y respectively
    for X, y in data_iter(batch_size, features, labels):
        with tf.GradientTape(persistent=True) as g:
            l = loss(net(X, w, b), y)    # Minibatch loss in X and y
            l = tf.reduce_mean(l)
        # Compute gradient on l with respect to [w,b]
        dl_dw = g.gradient(l, w)
        dl_db = g.gradient(l, b)
        sgd([w, b], [dl_dw, dl_db], lr, batch_size)     # Update parameters using their gradient
    train_l = loss(net(features, w, b), labels)
    print(f'epoch {epoch+1}, loss {float(tf.reduce_mean(train_l))}')
```

In this case, because we synthesized the data ourselves,
we know precisely what the true parameters are.
Thus, we can evaluate our success in training
by comparing the true parameters
with those that we learned through our training loop.
Indeed they turn out to be very close to each other.

```{.python .input}
print('Error in estimating w', true_w - w.reshape(true_w.shape))
print('Error in estimating b', true_b - b)
```

```{.python .input}
#@tab pytorch
print('Error in estimating w', true_w - w.reshape(true_w.shape))
print('Error in estimating b', true_b - b)
```

```{.python .input}
#@tab tensorflow
print('Error in estimating w', true_w - tf.reshape(w, true_w.shape))
print('Error in estimating b', true_b - b)
```

Note that we should not take it for granted
that we are able to recover the parameters accurately.
This only happens for a special category problems:
strongly convex optimization problems with "enough" data to ensure
that the noisy samples allow us to recover the underlying dependency.
In most cases this is *not* the case.
In fact, the parameters of a deep network
are rarely the same (or even close) between two different runs,
unless all conditions are identical,
including the order in which the data is traversed.
However, in machine learning, we are typically less concerned
with recovering true underlying parameters,
and more concerned with parameters that lead to accurate prediction.
Fortunately, even on difficult optimization problems,
stochastic gradient descent can often find remarkably good solutions,
owing partly to the fact that, for deep networks,
there exist many configurations of the parameters
that lead to accurate prediction.

## Summary

We saw how a deep network can be implemented
and optimized from scratch, using just `ndarray` and auto differentiation,
without any need for defining layers, fancy optimizers, etc.
This only scratches the surface of what is possible.
In the following sections, we will describe additional models
based on the concepts that we have just introduced
and learn how to implement them more concisely.

## Exercises

1. What would happen if we were to initialize the weights $\mathbf{w} = 0$. Would the algorithm still work?
1. Assume that you are
   [Georg Simon Ohm](https://en.wikipedia.org/wiki/Georg_Ohm) trying to come up
   with a model between voltage and current. Can you use auto differentiation to learn the parameters of your model.
1. Can you use [Planck's Law](https://en.wikipedia.org/wiki/Planck%27s_law) to determine the temperature of an object using spectral energy density?
1. What are the problems you might encounter if you wanted to  compute the second derivatives? How would you fix them?
1.  Why is the `reshape` function needed in the `squared_loss` function?
1. Experiment using different learning rates to find out how fast the loss function value drops.
1. If the number of examples cannot be divided by the batch size, what happens to the `data_iter` function's behavior?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/42)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/43)
:end_tab:
