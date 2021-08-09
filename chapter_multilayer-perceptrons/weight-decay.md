```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Weight Decay
:label:`sec_weight_decay`

Now that we have characterized the problem of overfitting,
we can introduce some standard techniques for regularizing models.
Recall that we can always mitigate overfitting
by going out and collecting more training data.
That can be costly, time consuming,
or entirely out of our control,
making it impossible in the short run.
For now, we can assume that we already have
as much high-quality data as our resources permit
and focus on regularization techniques.

Recall that in our
polynomial regression example
(:numref:`sec_model_selection`)
we could limit our model's capacity
simply by tweaking the degree
of the fitted polynomial.
Indeed, limiting the number of features
is a popular technique to mitigate overfitting.
However, simply tossing aside features
can be too blunt an instrument for the job.
Sticking with the polynomial regression
example, consider what might happen
with high-dimensional inputs.
The natural extensions of polynomials
to multivariate data are called *monomials*,
which are simply products of powers of variables.
The degree of a monomial is the sum of the powers.
For example, $x_1^2 x_2$, and $x_3 x_5^2$
are both monomials of degree 3.

Note that the number of terms with degree $d$
blows up rapidly as $d$ grows larger.
Given $k$ variables, the number of monomials
of degree $d$ (i.e., $k$ multichoose $d$) is ${k - 1 + d} \choose {k - 1}$.
Even small changes in degree, say from $2$ to $3$,
dramatically increase the complexity of our model.
Thus we often need a more fine-grained tool
for adjusting function complexity.


## Norms and Weight Decay

We have described
both the $\ell_2$ norm and the $\ell_1$ norm,
which are special cases of the more general $\ell_p$ norm
in :numref:`subsec_lin-algebra-norms`.
(***Weight decay* (commonly called $\ell_2$ regularization),
might be the most widely-used technique
for regularizing parametric machine learning models.**)
The technique is motivated by the basic intuition
that among all functions $f$,
the function $f = 0$
(assigning the value $0$ to all inputs)
is in some sense the *simplest*,
and that we can measure the complexity
of a function by its distance from zero.
But how precisely should we measure
the distance between a function and zero?
There is no single right answer.
In fact, entire branches of mathematics,
including parts of functional analysis
and the theory of Banach spaces,
are devoted to answering this issue.

One simple interpretation might be
to measure the complexity of a linear function
$f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$
by some norm of its weight vector, e.g., $\| \mathbf{w} \|^2$.
The most common method for ensuring a small weight vector
is to add its norm as a penalty term
to the problem of minimizing the loss.
Thus we replace our original objective,
*minimizing the prediction loss on the training labels*,
with new objective,
*minimizing the sum of the prediction loss and the penalty term*.
Now, if our weight vector grows too large,
our learning algorithm might focus
on minimizing the weight norm $\| \mathbf{w} \|^2$
vs. minimizing the training error.
That is exactly what we want.
To illustrate things in code,
let's revive our previous example
from :numref:`sec_linear_regression` for linear regression.
There, our loss was given by

$$L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

Recall that $\mathbf{x}^{(i)}$ are the features,
$y^{(i)}$ are labels for all data examples $i$, and $(\mathbf{w}, b)$
are the weight and bias parameters, respectively.
To penalize the size of the weight vector,
we must somehow add $\| \mathbf{w} \|^2$ to the loss function,
but how should the model trade off the
standard loss for this new additive penalty?
In practice, we characterize this tradeoff
via the *regularization constant* $\lambda$,
a non-negative hyperparameter
that we fit using validation data:

$$L(\mathbf{w}, b) + \frac{\lambda}{2} \|\mathbf{w}\|^2.$$

For $\lambda = 0$, we recover our original loss function.
For $\lambda > 0$, we restrict the size of $\| \mathbf{w} \|$.
We divide by $2$ by convention:
when we take the derivative of a quadratic function,
the $2$ and $1/2$ cancel out, ensuring that the expression
for the update looks nice and simple.
The astute reader might wonder why we work with the squared
norm and not the standard norm (i.e., the Euclidean distance).
We do this for computational convenience.
By squaring the $\ell_2$ norm, we remove the square root,
leaving the sum of squares of
each component of the weight vector.
This makes the derivative of the penalty easy to compute: the sum of derivatives equals the derivative of the sum.


Moreover, you might ask why we work with the $\ell_2$ norm
in the first place and not, say, the $\ell_1$ norm.
In fact, other choices are valid and
popular throughout statistics.
While $\ell_2$-regularized linear models constitute
the classic *ridge regression* algorithm,
$\ell_1$-regularized linear regression
is a similarly fundamental model in statistics, which is popularly known as *lasso regression*.


One reason to work with the $\ell_2$ norm
is that it places an outsize penalty
on large components of the weight vector.
This biases our learning algorithm
towards models that distribute weight evenly
across a larger number of features.
In practice, this might make them more robust
to measurement error in a single variable.
By contrast, $\ell_1$ penalties lead to models
that concentrate weights on a small set of features by clearing the other weights to zero.
This is called *feature selection*,
which may be desirable for other reasons.


Using the same notation in :eqref:`eq_linreg_batch_update`,
the minibatch stochastic gradient descent updates
for $\ell_2$-regularized regression follow:

$$
\begin{aligned}
\mathbf{w} & \leftarrow \left(1- \eta\lambda \right) \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right).
\end{aligned}
$$

As before, we update $\mathbf{w}$ based on the amount
by which our estimate differs from the observation.
However, we also shrink the size of $\mathbf{w}$ towards zero.
That is why the method is sometimes called "weight decay":
given the penalty term alone,
our optimization algorithm *decays*
the weight at each step of training.
In contrast to feature selection,
weight decay offers us a continuous mechanism
for adjusting the complexity of a function.
Smaller values of $\lambda$ correspond
to less constrained $\mathbf{w}$,
whereas larger values of $\lambda$
constrain $\mathbf{w}$ more considerably.

Whether we include a corresponding bias penalty $b^2$
can vary across implementations,
and may vary across layers of a neural network.
Often, we do not regularize the bias term
of a network's output layer.

## High-Dimensional Linear Regression

We can illustrate the benefits of
weight decay
through a simple synthetic example.

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

First, we [**generate some data as before**]

(**$$y = 0.05 + \sum_{i = 1}^d 0.01 x_i + \epsilon \text{ where }
\epsilon \sim \mathcal{N}(0, 0.01^2).$$**)

We choose our label to be a linear function of our inputs,
corrupted by Gaussian noise with zero mean and standard deviation 0.01.
To make the effects of overfitting pronounced,
we can increase the dimensionality of our problem to $d = 200$
and work with a small training set containing only 20 examples.

```{.python .input}
%%tab all
class Data(d2l.DataModule):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        self.save_hyperparameters()                
        n = num_train + num_val 
        if tab.selected('mxnet') or tab.selected('pytorch'):
            self.X = d2l.randn(n, num_inputs)
            noise = d2l.randn(n, 1) * 0.01
        if tab.selected('tensorflow'):
            self.X = d2l.normal((n, num_inputs))
            noise = d2l.normal((n, 1)) * 0.1
        w, b = d2l.ones((num_inputs, 1)) * 0.01, 0.05
        self.y = d2l.matmul(self.X, w) + b

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.X, self.y], train, i)
```

## Implementation from Scratch

In the following, we will implement weight decay from scratch,
simply by adding the squared $\ell_2$ penalty
to the original target function.

### (**Defining $\ell_2$ Norm Penalty**)

Perhaps the most convenient way to implement this penalty
is to square all terms in place and sum them up.

```{.python .input}
%%tab all
def l2_penalty(w):
    return d2l.reduce_sum(w**2) / 2
```

### Defining the Model

```{.python .input}
%%tab all
class WeightDecayScratch(d2l.LinearRegressionScratch):
    def __init__(self, num_inputs, lambd, lr, sigma=0.01):
        super().__init__(num_inputs, lr, sigma)
        self.save_hyperparameters()
        
    def loss(self, y_hat, y):
        return super().loss(y_hat, y) + self.lambd * l2_penalty(self.w)        
```

```{.python .input}
%%tab all
data = Data(num_train=20, num_val=100, num_inputs=200, batch_size=5)
trainer = d2l.Trainer(max_epochs=10)

def train_scratch(lambd):    
    model = WeightDecayScratch(num_inputs=200, lambd=lambd, lr=0.003)
    model.board.ylim = [1e-3, 1]
    trainer.fit(model, data)
    print('L2 norm of w:', float(l2_penalty(model.w)))
```

### [**Training without Regularization**]

We now run this code with `lambd = 0`,
disabling weight decay.
Note that we overfit badly,
decreasing the training error but not the
test error---a textbook case of overfitting.

```{.python .input}
%%tab all
train_scratch(0)
```

### [**Using Weight Decay**]

Below, we run with substantial weight decay.
Note that the training error increases
but the test error decreases.
This is precisely the effect
we expect from regularization.

```{.python .input}
%%tab all
train_scratch(3)
```

## [**Concise Implementation**]

Because weight decay is ubiquitous
in neural network optimization,
the deep learning framework makes it especially convenient,
integrating weight decay into the optimization algorithm itself
for easy use in combination with any loss function.
Moreover, this integration serves a computational benefit,
allowing implementation tricks to add weight decay to the algorithm,
without any additional computational overhead.
Since the weight decay portion of the update
depends only on the current value of each parameter,
the optimizer must touch each parameter once anyway.

:begin_tab:`mxnet`
In the following code, we specify
the weight decay hyperparameter directly
through `wd` when instantiating our `Trainer`.
By default, Gluon decays both
weights and biases simultaneously.
Note that the hyperparameter `wd`
will be multiplied by `wd_mult`
when updating model parameters.
Thus, if we set `wd_mult` to zero,
the bias parameter $b$ will not decay.
:end_tab:

:begin_tab:`pytorch`
In the following code, we specify
the weight decay hyperparameter directly
through `weight_decay` when instantiating our optimizer.
By default, PyTorch decays both
weights and biases simultaneously. Here we only set `weight_decay` for
the weight, so the bias parameter $b$ will not decay.
:end_tab:

:begin_tab:`tensorflow`
In the following code, we create an $\ell_2$ regularizer with
the weight decay hyperparameter `wd` and apply it to the layer
through the `kernel_regularizer` argument.
:end_tab:

```{.python .input}
%%tab mxnet
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        
    def configure_optimizers(self):
        self.collect_params('.*bias').setattr('wd_mult', 0)
        return gluon.Trainer(self.collect_params(),
                             'sgd', {'learning_rate': self.lr})
```

```{.python .input}
%%tab pytorch
class WeightDecay(d2l.LinearRegression):
    def __init__(self, num_inputs, wd, lr):
        super().__init__(num_inputs, lr)
        self.save_hyperparameters()
        
    def configure_optimizers(self):
        return torch.optim.SGD([
            {"params":self.net.weight,'weight_decay': self.wd},
            {"params":self.net.bias}], lr=self.lr)
```

```{.python .input}
%%tab tensorflow
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.net = tf.keras.layers.Dense(
            1, kernel_regularizer=tf.keras.regularizers.l2(wd))
        
    def loss(self, y_hat, y):
        return super().loss(y_hat, y) + self.net.losses
```

[**The plots look identical to those when
we implemented weight decay from scratch**].
However, they run appreciably faster
and are easier to implement,
a benefit that will become more
pronounced for larger problems.

```{.python .input}
%%tab all
if tab.selected('mxnet') or tab.selected('tensorflow'):    
    model = WeightDecay(wd=3, lr=0.003)
if tab.selected('pytorch'):
    model = WeightDecay(num_inputs=200, wd=3, lr=0.003)
    
model.board.ylim = [1e-3, 1]
trainer.fit(model, data)
print('L2 norm of w:', float(l2_penalty(model.get_w_b()[0]))) 
```

So far, we only touched upon one notion of
what constitutes a simple linear function.
Moreover, what constitutes a simple nonlinear function
can be an even more complex question.
For instance, [reproducing kernel Hilbert space (RKHS)](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space)
allows one to apply tools introduced
for linear functions in a nonlinear context.
Unfortunately, RKHS-based algorithms
tend to scale poorly to large, high-dimensional data.
In this book we will default to the simple heuristic
of applying weight decay on all layers of a deep network.

## Summary

* Regularization is a common method for dealing with overfitting. It adds a penalty term to the loss function on the training set to reduce the complexity of the learned model.
* One particular choice for keeping the model simple is weight decay using an $\ell_2$ penalty. This leads to weight decay in the update steps of the learning algorithm.
* The weight decay functionality is provided in optimizers from deep learning frameworks.
* Different sets of parameters can have different update behaviors within the same training loop.



## Exercises

1. Experiment with the value of $\lambda$ in the estimation problem in this section. Plot training and test accuracy as a function of $\lambda$. What do you observe?
1. Use a validation set to find the optimal value of $\lambda$. Is it really the optimal value? Does this matter?
1. What would the update equations look like if instead of $\|\mathbf{w}\|^2$ we used $\sum_i |w_i|$ as our penalty of choice ($\ell_1$ regularization)?
1. We know that $\|\mathbf{w}\|^2 = \mathbf{w}^\top \mathbf{w}$. Can you find a similar equation for matrices (see the Frobenius norm in :numref:`subsec_lin-algebra-norms`)?
1. Review the relationship between training error and generalization error. In addition to weight decay, increased training, and the use of a model of suitable complexity, what other ways can you think of to deal with overfitting?
1. In Bayesian statistics we use the product of prior and likelihood to arrive at a posterior via $P(w \mid x) \propto P(x \mid w) P(w)$. How can you identify $P(w)$ with regularization?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/98)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/99)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/236)
:end_tab:
