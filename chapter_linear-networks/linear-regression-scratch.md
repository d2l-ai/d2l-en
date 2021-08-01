# Linear Regression Implementation from Scratch
:label:`sec_linear_scratch`

Now that you understand the key ideas behind linear regression,
we can begin to work through a hands-on implementation in code.
In this section, (**we will implement the entire method from scratch,
including the data pipeline, the model,
the loss function, and the minibatch stochastic gradient descent optimizer.**)
While modern deep learning frameworks can automate nearly all of this work,
implementing things from scratch is the only way
to make sure that you really know what you are doing.
Moreover, when it comes time to customize models,
defining our own layers or loss functions,
understanding how things work under the hood will prove handy.
In this section, we will rely only on tensors and automatic differentiation.
For elegance of implementation we use a modular API inspired in its structure 
by [PyTorch Lightning](https://www.pytorchlightning.ai/). 
Later on, we will introduce a more concise implementation,
taking advantage of bells and whistles of deep learning frameworks while retaining 
the structure of what follows below.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
import random
npx.set_np()
```

```{.python .input  n=1}
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

To keep things simple, we [**construct an artificial dataset
according to a linear model with additive noise.**]
Our task will be to recover the model parameters
using the finite number of samples in our dataset. 
We keep the data low-dimensional so we can visualize it easily.
In the following code snippet, we generate 
1000 examples, each consisting of 2 features
drawn from a standard normal distribution.
Thus the matrix of features is
$\mathbf{X}\in \mathbb{R}^{1000 \times 2}$.

(**The true parameters generating our true targets are
$\mathbf{w} = [2, -3.4]^\top$ and $b = 4.2$.**) 
These are corrupted by additive noise $\epsilon$, drawn independently for each $\mathbf{x}$. 
As such, the synthetic labels satisfy: 

(**$$\mathbf{y}= \mathbf{X} \mathbf{w} + b + \mathbf\epsilon.$$**)

For convenience we assume that $\epsilon$ arises from a normal distribution with mean of 0.
To make our problem easy, we will set its standard deviation to $\sigma = 0.01$.
The following code generates the synthetic dataset.

```{.python .input}
#@tab mxnet, pytorch
class SyntheticRegressionData(d2l.DataModule):  #@save
    def __init__(self, w, b, num_examples=1000, batch_size=8):
        super().__init__()
        self.save_hyperparameters()
        self.X = d2l.normal(0, 1, (num_examples, len(w)))
        y = d2l.matmul(self.X, w) + b + d2l.normal(0, 0.01, (num_examples,))
        self.y = d2l.reshape(y, (-1, 1))
```

```{.python .input}
#@tab tensorflow
class SyntheticRegressionData(d2l.DataModule):  #@save
    def __init__(self, w, b, num_examples=1000, batch_size=8):
        super().__init__()
        self.save_hyperparameters()
        self.X = tf.random.normal((num_examples, w.shape[0]))
        y = d2l.matmul(self.X, tf.reshape(w, (-1, 1))) + b
        self.y = y + tf.random.normal(y.shape, stddev=0.01)
```

```{.python .input}
#@tab all
true_w = d2l.tensor([2, -3.4])
true_b = 4.2
data = SyntheticRegressionData(true_w, true_b)
```

[**Each row in `features` consists of a vector in $\mathbb{R}^2$ and each row in `labels` is a scalar.**]

```{.python .input}
#@tab all
print('features:', data.X[0],'\nlabel:', data.y[0])
```

## Reading the Dataset

Training models consists of
making multiple passes over the dataset,
grabbing one minibatch of examples at a time,
and using them to update our model.
Since this process is so fundamental
to training machine learning algorithms,
it is worth defining a utility function
to shuffle the dataset and access it in minibatches.

In the following code, we [**define the `data_iter` function**] (~~that~~)
to demonstrate one possible implementation of this functionality.
It (**takes a batch size, a matrix of features,
and a vector of labels, yielding minibatches of size `batch_size`.**)
Each minibatch consists of a tuple of features and labels.

```{.python .input}
#@tab mxnet, pytorch
@d2l.add_to_class(SyntheticRegressionData)
def train_dataloader(self):
    indices = list(range(self.num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, self.num_examples, self.batch_size):
        batch_indices = d2l.tensor(
            indices[i: min(i + self.batch_size, self.num_examples)])
        yield self.X[batch_indices], self.y[batch_indices]
```

```{.python .input}
#@tab tensorflow
@d2l.add_to_class(SyntheticRegressionData)
def train_dataloader(self):
    indices = list(range(self.num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, self.num_examples, self.batch_size):
        j = tf.constant(indices[
            i : min(i+self.batch_size, self.num_examples)])
        yield tf.gather(self.X, j), tf.gather(self.y, j)
```

In general we want to use reasonably sized minibatches
to take advantage of the GPU, as it excels at parallel computation. 
Since examples can be fed through our models in parallel
and the gradient of the loss function for each example can also be taken in parallel,
GPUs allow us to process hundreds of examples in scarcely more time
than it might take to process just a single example.

To build some intuition, let us inspect the first minibatch of 
data. Each minibatch of features provides us both with its size and the dimensionality of input features. 
Likewise, our minibatch of labels will have a matching shape given by `batch_size`.

```{.python .input}
#@tab all
X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '\ny shape:', y.shape)
```

While seemingly quite innocuous, the invocation of `iter(data.train_dataloader())` 
illustrates the power of Python's object oriented design. We ended up adding 
a method to the `SyntheticRegressionData` class *after* creating the `data` 
object. Nonetheless, the object benefits from the a-posteriori addition of 
functionality to the class. 

As we run the iteration, we obtain distinct minibatches
until the entire dataset has been exhausted (try this).
While the iteration implemented above is good for didactic purposes,
it is inefficient in ways that might get us in trouble on real problems.
For example, it requires that we load all the data in memory
and that we perform lots of random memory access.
The built-in iterators implemented in a deep learning framework
are considerably more efficient and they can deal
with sources such as data stored in files, data received via a stream, or data 
generated/processed on the fly.

## Defining the Model

[**Before we can begin optimizing our model's parameters**] by minibatch SGD,
(**we need to have some parameters in the first place.**)
In the following we initialize weights by drawing
random numbers from a normal distribution with mean 0
and a standard deviation of 0.01. Moreover we set the bias to 0.

```{.python .input}
class LinearRegressionScratch(d2l.Module):  #@save
    def __init__(self, num_inputs, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.w = d2l.normal(0, 0.01, size=(num_inputs, num_outputs))
        self.b = d2l.zeros(num_outputs)
        self.w.attach_grad()
        self.b.attach_grad()
```

```{.python .input}
#@tab pytorch
class LinearRegressionScratch(d2l.Module):
    def __init__(self, num_inputs, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.w = torch.normal(0, 0.01, size=(num_inputs, num_outputs),
                              requires_grad=True)
        self.b = torch.zeros(num_outputs, requires_grad=True)
```

```{.python .input}
#@tab tensorflow
class LinearRegressionScratch(d2l.Module):
    def __init__(self, num_inputs, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        w = tf.random.normal((num_inputs, num_outputs), 0, 0.01)
        b = tf.zeros(num_outputs)
        self.w = tf.Variable(w, trainable=True)
        self.b = tf.Variable(b, trainable=True)
```

After initializing our parameters,
our next task is to update them until
they fit the data sufficiently well.
Each update requires taking the gradient
of our loss function with respect to the parameters.
Given this gradient, we can update each parameter
in the direction that may reduce the loss.

Since nobody wants to compute gradients explicitly
(this is tedious and error prone),
we use automatic differentiation,
as introduced in :numref:`sec_autograd`.

Next, we must [**define our model,
relating its inputs and parameters to its outputs.**]
For our linear model we simply take the matrix-vector product
of the input features $\mathbf{X}$ and the model weights $\mathbf{w}$,
and add the offset $b$ to each example.
$\mathbf{Xw}$ is a vector and $b$ is a scalar.
Due to the broadcasting mechanism of :numref:`subsec_broadcasting`,
when we add a vector and a scalar,
the scalar is added to each component of the vector.

```{.python .input}
#@tab mxnet, pytorch
@d2l.add_to_class(LinearRegressionScratch)
def forward(self, X):
    """The linear regression model."""
    return d2l.matmul(X, self.w) + self.b
```

```{.python .input}
#@tab tensorflow
@d2l.add_to_class(LinearRegressionScratch)
def call(self, X):
    """The linear regression model."""
    return d2l.matmul(X, self.w) + self.b
```

## Defining the Loss Function

Since [**updating our model requires taking
the gradient of our loss function,**]
we ought to (**define the loss function first.**)
Here we use the squared loss function
of :numref:`sec_linear_regression`.
In the implementation, we need to transform the true value `y`
into the predicted value's shape `y_hat`.
The result returned by the following function
will also have the same shape as `y_hat`.

```{.python .input}
#@tab all
def mse(y_hat, y):  #@save
    """Squared loss."""
    loss = (y_hat - d2l.reshape(y, y_hat.shape)) ** 2 / 2
    return d2l.reduce_mean(loss)
```

```{.python .input}
#@tab all
@d2l.add_to_class(LinearRegressionScratch)
def training_step(self, batch, batch_idx):
    X, y = batch
    l = mse(self(X), y)
    self.board.draw({'step':batch_idx, 'loss':l}, every_n=10)
    return l
```

## Defining the Optimization Algorithm

As discussed in :numref:`sec_linear_regression`,
linear regression has a closed-form solution.
However, our goal here is to illustrate how to
use minibatch SGD. Hence we will take this opportunity
to introduce your first working example of SGD.
At each step, using a minibatch randomly drawn from our dataset,
we estimate the gradient of the loss with respect to the parameters.
Next, we update the parameters
in the direction that may reduce the loss.

The following code applies the update, given a set of parameters, a learning rate `lr`, and a batch size.
Since the loss is calculated as a sum over the minibatch 
we normalize the step size by the `batch_size`,
so that the magnitude of a typical step size
does not depend on our choice of the batch size. 
In later chapters we will investigate how learning rates should be adjusted 
for very large minibatches as they arise in distributed large scale learning. 
For now, though, we can ignore this dependency. 

```{.python .input}
class SGD(d2l.HyperParameters):  #@save
    def __init__(self, params, lr):
        """Minibatch stochastic gradient descent."""
        self.save_hyperparameters()

    def step(self):
        for param in self.params:
            param -= self.lr * param.grad
```

```{.python .input}
#@tab pytorch
class SGD(d2l.HyperParameters):  #@save
    def __init__(self, params, lr):
        """Minibatch stochastic gradient descent."""
        self.save_hyperparameters()

    def step(self):
        for param in self.params:
            param -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
```

```{.python .input}
#@tab mxnet, pytorch
@d2l.add_to_class(LinearRegressionScratch)
def configure_optimizers(self):
    return SGD([self.w, self.b], self.lr)
```

```{.python .input}
#@tab tensorflow
class SGD(d2l.HyperParameters):  #@save
    def __init__(self, lr):
        """Minibatch stochastic gradient descent."""
        self.save_hyperparameters()

    def apply_gradients(self, grads_and_vars):
        for grad, param in grads_and_vars:
            param.assign_sub(self.lr * grad)
```

```{.python .input}
#@tab tensorflow
@d2l.add_to_class(LinearRegressionScratch)
def configure_optimizers(self):
    return SGD(self.lr)
```

## Training

Now that we have all of the parts in place (parameters, data, loss function, model, and optimizer),
we are ready to [**implement the main training loop.**]
It is crucial that you understand this code well 
since it the archetype of almost all training loops in deep learning.

In each iteration, we grab a minibatch of training examples,
and pass them through our model to obtain a set of predictions.
After calculating the loss, we initiate the backwards pass through the network,
storing the gradients with respect to each parameter.
Finally, we will call the optimization algorithm `sgd`
to update the model parameters. In summary, we will execute the following loop:

* Initialize parameters $(\mathbf{w}, b)$
* Repeat until done
    * Compute gradient $\mathbf{g} \leftarrow \partial_{(\mathbf{w},b)} \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} l(\mathbf{x}^{(i)}, y^{(i)}, \mathbf{w}, b)$
    * Update parameters $(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta \mathbf{g}$

In each *epoch*,
we will iterate through the entire dataset, 
using the `data_iter` function, passing once
through every example in the training set
(assuming that the number of examples is divisible by the batch size).
The number of epochs `num_epochs` and the learning rate `lr` are both hyperparameters,
which we set here to 3 and 0.03, respectively.
In general, setting hyperparameters is tricky
and requires some adjustment for different problems and 
network architectures. We elide these details for now but revise them 
later in :numref:`chap_optimization`.

```{.python .input}
#@tab pytorch
@d2l.add_to_class(d2l.Trainer)
def fit(self, model, data):
    train_dataloader = data.train_dataloader()
    self.train_batch_idx = 0
    optim = model.configure_optimizers()
    for epoch in range(self.max_epochs):
        for batch in train_dataloader:
            loss = model.training_step(batch, self.train_batch_idx)
            optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                optim.step()
            self.train_batch_idx += 1
```

```{.python .input}
#@tab mxnet
@d2l.add_to_class(d2l.Trainer)
def fit(self, model, data):
    train_dataloader = data.train_dataloader()
    self.train_batch_idx = 0
    optim = model.configure_optimizers()
    for epoch in range(self.max_epochs):
        for batch in train_dataloader:
            with autograd.record():
                loss = model.training_step(batch, self.train_batch_idx)
            loss.backward()
            optim.step()
            self.train_batch_idx += 1
```

```{.python .input}
#@tab tensorflow
@d2l.add_to_class(d2l.Trainer)
def fit(self, model, data):
    train_dataloader = data.train_dataloader()
    self.train_batch_idx = 0
    optim = model.configure_optimizers()
    for epoch in range(self.max_epochs):
        for batch in train_dataloader:
            with tf.GradientTape() as tape:
                loss = model.training_step(batch, self.train_batch_idx)
            grads = tape.gradient(loss, model.trainable_variables)
            optim.apply_gradients(zip(grads, model.trainable_variables))
            self.train_batch_idx += 1
```

Note the general pattern of taking a minibatch `batch`, computing the `loss`, followed by a gradient computation and lastly, the application of the gradient to the set of model parameters via an optimizer `optim`. The majority of deep learning training algorithms follows this pattern. Now that we defined the training loop, let's use it. 

```{.python .input}
#@tab all
model = LinearRegressionScratch(2, 1, lr=0.03)
trainer = d2l.Trainer(3)
trainer.fit(model, data)
```

In this case, because we synthesized the dataset ourselves,
we know precisely what the true parameters are.
Thus, we can [**evaluate our success in training
by comparing the true parameters
with those that we learned**] through our training loop.
Indeed they turn out to be very close to each other.

```{.python .input}
#@tab all
print(f'error in estimating w: {data.w - d2l.reshape(model.w, data.w.shape)}')
print(f'error in estimating b: {data.b - model.b}')
```

Note that we should not take it for granted
that we are able to recover the parameters perfectly.
However, in machine learning, we are typically less concerned
with recovering true underlying parameters,
and more concerned with parameters that lead to highly accurate prediction :cite:`vapnik1992principles`.
Fortunately, even on difficult optimization problems,
stochastic gradient descent can often find remarkably good solutions,
owing partly to the fact that, for deep networks,
there exist many configurations of the parameters
that lead to highly accurate prediction. Likewise, the stochastic nature of 
optimization ensures that are less likely to get 'stuck' in shallow local minima. 


## Summary

In this section we took a significant step forward towards designing deep learning systems by implementing
the the canonical structure of a training loop. In this process we built a data loader, a model, a loss function, an optimization procedure and a visualization and monitoring tool. We did this by composing a Python object that contains all relevant parts for training a model. While none of what we implemented is efficient, it is still sufficient to solve a small toy problem quickly. In the next sections we will see how to do this a) more concisely and b) more efficiently such that we can use the GPUs in our computer to their full extent. Once we are comfortable with that, we graduate to more advanced network architectures, specific datasets, and optimizers in the subsequent chapters. 


## Exercises

1. What would happen if we were to initialize the weights to zero. Would the algorithm still work? What if we 
   initialized the parameters with variance $1,000$ rather than $0.01$?
1. Assume that you are [Georg Simon Ohm](https://en.wikipedia.org/wiki/Georg_Ohm) trying to come up
   with a model for resistors that relates voltage and current. Can you use automatic 
   differentiation to learn the parameters of your model?
1. Can you use [Planck's Law](https://en.wikipedia.org/wiki/Planck%27s_law) to determine the temperature of an object 
   using spectral energy density? For reference, the spectral density $B$ of radiation emanating from a black body is
   $B(\lambda, T) = \frac{2 hc^2}{\lambda^5} \cdot \left(\exp \frac{h c}{\lambda k T} - 1\right)^{-1}$. Here 
   $\lambda$ is the wavelength, $T$ the temperature, $c$ the speed of light, $h$ Planck's quantum, and $k$ the 
   Boltzmann constant. You measure the energy for different wavelengths $\lambda$ and you now need to fit the spectral 
   density curve to Planck's law. 
1. What are the problems you might encounter if you wanted to compute the second derivatives of the loss? How would 
   you fix them? 
1. Why is the `reshape` function needed in the `squared_loss` function?
1. Experiment using different learning rates to find out how quickly the loss function value drops. Can you reduce the 
   error by increasing the number of epochs of training?
1. If the number of examples cannot be divided by the batch size, what happens to `data_iter` at the end of an epoch?
1. Try implementing a different loss function, such as the absolute value loss `(y_hat - d2l.reshape(y, y_hat.shape)).abs().sum()`.
    1. Check what happens for regular data.
    1. Check whether there is a difference in behavior if you actively perturb some entries of $\mathbf{y}$, 
       such as $y_5 = 10,000$. 
    1. Can you think of a cheap solution for combining the best aspects of squared loss and absolute value loss? 
       Hint: how can you avoid really large gradient values?
1. Why do we need to reshuffle the dataset? Can you design a case where a maliciously dataset would break the 
   optimization algorithm otherwise?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/42)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/43)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/201)
:end_tab:

TODO, remove the below deprecated code
