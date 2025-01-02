```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Linear Regression Implementation from Scratch
:label:`sec_linear_scratch`

We are now ready to work through 
a fully functioning implementation 
of linear regression. 
In this section, 
(**we will implement the entire method from scratch,
including (i) the model; (ii) the loss function;
(iii) a minibatch stochastic gradient descent optimizer;
and (iv) the training function 
that stitches all of these pieces together.**)
Finally, we will run our synthetic data generator
from :numref:`sec_synthetic-regression-data`
and apply our model
on the resulting dataset. 
While modern deep learning frameworks 
can automate nearly all of this work,
implementing things from scratch is the only way
to make sure that you really know what you are doing.
Moreover, when it is time to customize models,
defining our own layers or loss functions,
understanding how things work under the hood will prove handy.
In this section, we will rely only 
on tensors and automatic differentiation.
Later, we will introduce a more concise implementation,
taking advantage of the bells and whistles of deep learning frameworks 
while retaining the structure of what follows below.

```{.python .input  n=2}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input  n=4}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input  n=5}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
import optax
```

## Defining the Model

[**Before we can begin optimizing our model's parameters**] by minibatch SGD,
(**we need to have some parameters in the first place.**)
In the following we initialize weights by drawing
random numbers from a normal distribution with mean 0
and a standard deviation of 0.01. 
The magic number 0.01 often works well in practice, 
but you can specify a different value 
through the argument `sigma`.
Moreover we set the bias to 0.
Note that for object-oriented design
we add the code to the `__init__` method of a subclass of `d2l.Module` (introduced in :numref:`subsec_oo-design-models`).

```{.python .input  n=6}
%%tab pytorch, mxnet, tensorflow
class LinearRegressionScratch(d2l.Module):  #@save
    """The linear regression model implemented from scratch."""
    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.w = d2l.normal(0, sigma, (num_inputs, 1))
            self.b = d2l.zeros(1)
            self.w.attach_grad()
            self.b.attach_grad()
        if tab.selected('pytorch'):
            self.w = d2l.normal(0, sigma, (num_inputs, 1), requires_grad=True)
            self.b = d2l.zeros(1, requires_grad=True)
        if tab.selected('tensorflow'):
            w = tf.random.normal((num_inputs, 1), mean=0, stddev=0.01)
            b = tf.zeros(1)
            self.w = tf.Variable(w, trainable=True)
            self.b = tf.Variable(b, trainable=True)
```

```{.python .input  n=7}
%%tab jax
class LinearRegressionScratch(d2l.Module):  #@save
    """The linear regression model implemented from scratch."""
    num_inputs: int
    lr: float
    sigma: float = 0.01

    def setup(self):
        self.w = self.param('w', nn.initializers.normal(self.sigma),
                            (self.num_inputs, 1))
        self.b = self.param('b', nn.initializers.zeros, (1))
```

Next we must [**define our model,
relating its input and parameters to its output.**]
Using the same notation as :eqref:`eq_linreg-y-vec`
for our linear model we simply take the matrix--vector product
of the input features $\mathbf{X}$ 
and the model weights $\mathbf{w}$,
and add the offset $b$ to each example.
The product $\mathbf{Xw}$ is a vector and $b$ is a scalar.
Because of the broadcasting mechanism 
(see :numref:`subsec_broadcasting`),
when we add a vector and a scalar,
the scalar is added to each component of the vector.
The resulting `forward` method 
is registered in the `LinearRegressionScratch` class
via `add_to_class` (introduced in :numref:`oo-design-utilities`).

```{.python .input  n=8}
%%tab all
@d2l.add_to_class(LinearRegressionScratch)  #@save
def forward(self, X):
    return d2l.matmul(X, self.w) + self.b
```

## Defining the Loss Function

Since [**updating our model requires taking
the gradient of our loss function,**]
we ought to (**define the loss function first.**)
Here we use the squared loss function
in :eqref:`eq_mse`.
In the implementation, we need to transform the true value `y`
into the predicted value's shape `y_hat`.
The result returned by the following method
will also have the same shape as `y_hat`. 
We also return the averaged loss value
among all examples in the minibatch.

```{.python .input  n=9}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LinearRegressionScratch)  #@save
def loss(self, y_hat, y):
    l = (y_hat - y) ** 2 / 2
    return d2l.reduce_mean(l)
```

```{.python .input  n=10}
%%tab jax
@d2l.add_to_class(LinearRegressionScratch)  #@save
def loss(self, params, X, y, state):
    y_hat = state.apply_fn({'params': params}, *X)  # X unpacked from a tuple
    l = (y_hat - d2l.reshape(y, y_hat.shape)) ** 2 / 2
    return d2l.reduce_mean(l)
```

## Defining the Optimization Algorithm

As discussed in :numref:`sec_linear_regression`,
linear regression has a closed-form solution.
However, our goal here is to illustrate 
how to train more general neural networks,
and that requires that we teach you 
how to use minibatch SGD.
Hence we will take this opportunity
to introduce your first working example of SGD.
At each step, using a minibatch 
randomly drawn from our dataset,
we estimate the gradient of the loss
with respect to the parameters.
Next, we update the parameters
in the direction that may reduce the loss.

The following code applies the update, 
given a set of parameters, a learning rate `lr`.
Since our loss is computed as an average over the minibatch, 
we do not need to adjust the learning rate against the batch size. 
In later chapters we will investigate 
how learning rates should be adjusted
for very large minibatches as they arise 
in distributed large-scale learning.
For now, we can ignore this dependency.

:begin_tab:`mxnet`
We define our `SGD` class, 
a subclass of `d2l.HyperParameters` (introduced in :numref:`oo-design-utilities`),
to have a similar API
as the built-in SGD optimizer.
We update the parameters in the `step` method.
It accepts a `batch_size` argument that can be ignored.
:end_tab:

:begin_tab:`pytorch`
We define our `SGD` class,
a subclass of `d2l.HyperParameters` (introduced in :numref:`oo-design-utilities`),
to have a similar API 
as the built-in SGD optimizer.
We update the parameters in the `step` method.
The `zero_grad` method sets all gradients to 0,
which must be run before a backpropagation step.
:end_tab:

:begin_tab:`tensorflow`
We define our `SGD` class,
a subclass of `d2l.HyperParameters` (introduced in :numref:`oo-design-utilities`),
to have a similar API
as the built-in SGD optimizer.
We update the parameters in the `apply_gradients` method.
It accepts a list of parameter and gradient pairs.
:end_tab:

```{.python .input  n=11}
%%tab mxnet, pytorch
class SGD(d2l.HyperParameters):  #@save
    """Minibatch stochastic gradient descent."""
    def __init__(self, params, lr):
        self.save_hyperparameters()

    if tab.selected('mxnet'):
        def step(self, _):
            for param in self.params:
                param -= self.lr * param.grad

    if tab.selected('pytorch'):
        def step(self):
            for param in self.params:
                param -= self.lr * param.grad

        def zero_grad(self):
            for param in self.params:
                if param.grad is not None:
                    param.grad.zero_()
```

```{.python .input  n=12}
%%tab tensorflow
class SGD(d2l.HyperParameters):  #@save
    """Minibatch stochastic gradient descent."""
    def __init__(self, lr):
        self.save_hyperparameters()

    def apply_gradients(self, grads_and_vars):
        for grad, param in grads_and_vars:
            param.assign_sub(self.lr * grad)
```

```{.python .input  n=13}
%%tab jax
class SGD(d2l.HyperParameters):  #@save
    """Minibatch stochastic gradient descent."""
    # The key transformation of Optax is the GradientTransformation
    # defined by two methods, the init and the update.
    # The init initializes the state and the update transforms the gradients.
    # https://github.com/deepmind/optax/blob/master/optax/_src/transform.py
    def __init__(self, lr):
        self.save_hyperparameters()

    def init(self, params):
        # Delete unused params
        del params
        return optax.EmptyState

    def update(self, updates, state, params=None):
        del params
        # When state.apply_gradients method is called to update flax's
        # train_state object, it internally calls optax.apply_updates method
        # adding the params to the update equation defined below.
        updates = jax.tree_util.tree_map(lambda g: -self.lr * g, updates)
        return updates, state

    def __call__():
        return optax.GradientTransformation(self.init, self.update)
```

We next define the `configure_optimizers` method, which returns an instance of the `SGD` class.

```{.python .input  n=14}
%%tab all
@d2l.add_to_class(LinearRegressionScratch)  #@save
def configure_optimizers(self):
    if tab.selected('mxnet') or tab.selected('pytorch'):
        return SGD([self.w, self.b], self.lr)
    if tab.selected('tensorflow', 'jax'):
        return SGD(self.lr)
```

## Training

Now that we have all of the parts in place
(parameters, loss function, model, and optimizer),
we are ready to [**implement the main training loop.**]
It is crucial that you understand this code fully
since you will employ similar training loops
for every other deep learning model
covered in this book.
In each *epoch*, we iterate through 
the entire training dataset, 
passing once through every example
(assuming that the number of examples 
is divisible by the batch size). 
In each *iteration*, we grab a minibatch of training examples,
and compute its loss through the model's `training_step` method. 
Then we compute the gradients with respect to each parameter. 
Finally, we will call the optimization algorithm
to update the model parameters. 
In summary, we will execute the following loop:

* Initialize parameters $(\mathbf{w}, b)$
* Repeat until done
    * Compute gradient $\mathbf{g} \leftarrow \partial_{(\mathbf{w},b)} \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} l(\mathbf{x}^{(i)}, y^{(i)}, \mathbf{w}, b)$
    * Update parameters $(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta \mathbf{g}$
 
Recall that the synthetic regression dataset 
that we generated in :numref:``sec_synthetic-regression-data`` 
does not provide a validation dataset. 
In most cases, however, 
we will want a validation dataset 
to measure our model quality. 
Here we pass the validation dataloader 
once in each epoch to measure the model performance.
Following our object-oriented design,
the `prepare_batch` and `fit_epoch` methods
are registered in the `d2l.Trainer` class
(introduced in :numref:`oo-design-training`).

```{.python .input  n=15}
%%tab all    
@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_batch(self, batch):
    return batch
```

```{.python .input  n=16}
%%tab pytorch
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    self.model.training = True
    for batch in self.train_dataloader:        
        loss = self.model.training_step(self.prepare_batch(batch))
        self.optim.zero_grad()
        with torch.no_grad():
            loss.backward()
            if self.gradient_clip_val > 0:  # To be discussed later
                self.clip_gradients(self.gradient_clip_val, self.model)
            self.optim.step()
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    self.model.training = False
    for batch in self.val_dataloader:
        with torch.no_grad():            
            self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
```

```{.python .input  n=17}
%%tab mxnet
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    for batch in self.train_dataloader:
        with autograd.record():
            loss = self.model.training_step(self.prepare_batch(batch))
        loss.backward()
        if self.gradient_clip_val > 0:
            self.clip_gradients(self.gradient_clip_val, self.model)
        self.optim.step(1)
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    for batch in self.val_dataloader:        
        self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
```

```{.python .input  n=18}
%%tab tensorflow
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    self.model.training = True
    for batch in self.train_dataloader:            
        with tf.GradientTape() as tape:
            loss = self.model.training_step(self.prepare_batch(batch))
        grads = tape.gradient(loss, self.model.trainable_variables)
        if self.gradient_clip_val > 0:
            grads = self.clip_gradients(self.gradient_clip_val, grads)
        self.optim.apply_gradients(zip(grads, self.model.trainable_variables))
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    self.model.training = False
    for batch in self.val_dataloader:        
        self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
```

```{.python .input  n=19}
%%tab jax
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    self.model.training = True
    if self.state.batch_stats:
        # Mutable states will be used later (e.g., for batch norm)
        for batch in self.train_dataloader:
            (_, mutated_vars), grads = self.model.training_step(self.state.params,
                                                           self.prepare_batch(batch),
                                                           self.state)
            self.state = self.state.apply_gradients(grads=grads)
            # Can be ignored for models without Dropout Layers
            self.state = self.state.replace(
                dropout_rng=jax.random.split(self.state.dropout_rng)[0])
            self.state = self.state.replace(batch_stats=mutated_vars['batch_stats'])
            self.train_batch_idx += 1
    else:
        for batch in self.train_dataloader:
            _, grads = self.model.training_step(self.state.params,
                                                self.prepare_batch(batch),
                                                self.state)
            self.state = self.state.apply_gradients(grads=grads)
            # Can be ignored for models without Dropout Layers
            self.state = self.state.replace(
                dropout_rng=jax.random.split(self.state.dropout_rng)[0])
            self.train_batch_idx += 1

    if self.val_dataloader is None:
        return
    self.model.training = False
    for batch in self.val_dataloader:
        self.model.validation_step(self.state.params,
                                   self.prepare_batch(batch),
                                   self.state)
        self.val_batch_idx += 1
```

We are almost ready to train the model,
but first we need some training data.
Here we use the `SyntheticRegressionData` class 
and pass in some ground truth parameters.
Then we train our model with 
the learning rate `lr=0.03` 
and set `max_epochs=3`. 
Note that in general, both the number of epochs 
and the learning rate are hyperparameters.
In general, setting hyperparameters is tricky
and we will usually want to use a three-way split,
one set for training, 
a second for hyperparameter selection,
and the third reserved for the final evaluation.
We elide these details for now but will revise them
later.

```{.python .input  n=20}
%%tab all
model = LinearRegressionScratch(2, lr=0.03)
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)
```

Because we synthesized the dataset ourselves,
we know precisely what the true parameters are.
Thus, we can [**evaluate our success in training
by comparing the true parameters
with those that we learned**] through our training loop.
Indeed they turn out to be very close to each other.

```{.python .input  n=21}
%%tab pytorch
with torch.no_grad():
    print(f'error in estimating w: {data.w - d2l.reshape(model.w, data.w.shape)}')
    print(f'error in estimating b: {data.b - model.b}')
```

```{.python .input  n=22}
%%tab mxnet, tensorflow
print(f'error in estimating w: {data.w - d2l.reshape(model.w, data.w.shape)}')
print(f'error in estimating b: {data.b - model.b}')
```

```{.python .input  n=23}
%%tab jax
params = trainer.state.params
print(f"error in estimating w: {data.w - d2l.reshape(params['w'], data.w.shape)}")
print(f"error in estimating b: {data.b - params['b']}")
```

We should not take the ability to exactly recover 
the ground truth parameters for granted.
In general, for deep models unique solutions
for the parameters do not exist,
and even for linear models,
exactly recovering the parameters
is only possible when no feature 
is linearly dependent on the others.
However, in machine learning, 
we are often less concerned
with recovering true underlying parameters,
but rather with parameters 
that lead to highly accurate prediction :cite:`Vapnik.1992`.
Fortunately, even on difficult optimization problems,
stochastic gradient descent can often find remarkably good solutions,
owing partly to the fact that, for deep networks,
there exist many configurations of the parameters
that lead to highly accurate prediction.


## Summary

In this section, we took a significant step 
towards designing deep learning systems 
by implementing a fully functional 
neural network model and training loop.
In this process, we built a data loader, 
a model, a loss function, an optimization procedure,
and a visualization and monitoring tool. 
We did this by composing a Python object 
that contains all relevant components for training a model. 
While this is not yet a professional-grade implementation
it is perfectly functional and code like this 
could already help you to solve small problems quickly.
In the coming sections, we will see how to do this
both *more concisely* (avoiding boilerplate code)
and *more efficiently* (using our GPUs to their full potential).



## Exercises

1. What would happen if we were to initialize the weights to zero. Would the algorithm still work? What if we
   initialized the parameters with variance $1000$ rather than $0.01$?
1. Assume that you are [Georg Simon Ohm](https://en.wikipedia.org/wiki/Georg_Ohm) trying to come up
   with a model for resistance that relates voltage and current. Can you use automatic
   differentiation to learn the parameters of your model?
1. Can you use [Planck's Law](https://en.wikipedia.org/wiki/Planck%27s_law) to determine the temperature of an object
   using spectral energy density? For reference, the spectral density $B$ of radiation emanating from a black body is
   $B(\lambda, T) = \frac{2 hc^2}{\lambda^5} \cdot \left(\exp \frac{h c}{\lambda k T} - 1\right)^{-1}$. Here
   $\lambda$ is the wavelength, $T$ is the temperature, $c$ is the speed of light, $h$ is Planck's constant, and $k$ is the
   Boltzmann constant. You measure the energy for different wavelengths $\lambda$ and you now need to fit the spectral
   density curve to Planck's law.
1. What are the problems you might encounter if you wanted to compute the second derivatives of the loss? How would
   you fix them?
1. Why is the `reshape` method needed in the `loss` function?
1. Experiment using different learning rates to find out how quickly the loss function value drops. Can you reduce the
   error by increasing the number of epochs of training?
1. If the number of examples cannot be divided by the batch size, what happens to `data_iter` at the end of an epoch?
1. Try implementing a different loss function, such as the absolute value loss `(y_hat - d2l.reshape(y, y_hat.shape)).abs().sum()`.
    1. Check what happens for regular data.
    1. Check whether there is a difference in behavior if you actively perturb some entries, such as $y_5 = 10000$, of $\mathbf{y}$.
    1. Can you think of a cheap solution for combining the best aspects of squared loss and absolute value loss?
       Hint: how can you avoid really large gradient values?
1. Why do we need to reshuffle the dataset? Can you design a case where a maliciously constructed dataset would break the optimization algorithm otherwise?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/42)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/43)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/201)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17976)
:end_tab:
