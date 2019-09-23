# Automatic Differentiation
:label:`chapter_autograd`

In machine learning, we *train* models, updating them successively 
so that they get better and better as they see more and more data. 
Usually, *getting better* means minimizing a *loss function*, 
a score that answers the question "how *bad* is our model?"
This question is more subtle than it appears.
Ultimately, what we really care about 
is producing a model that performs well on data 
that we have never seen before.
But we can only fit the model to data that we can actually see.
Thus we can decompose the task of fitting models into two key concerns:
*optimization* the process of fitting our models to observed data
and *generalization* the mathematical principles and practitioners wisdom
that guide as to how to produce models whose validity extends 
beyond the exact set of datapoints used to train it. 

This section addresses the calculation of derivatives,
a crucial step in nearly all deep learning optimization algorithms.
With neural networks, we typically choose loss functions 
that are differentiable with respect to our model's parameters.
Put simply, this means that for each parameter, 
we can determine how rapidly the loss would increase or decrease,
were we to *increase* or *decrease* that parameter
by an infinitessimally small amount.
While the calculations for taking these derivatives are straightforward,
requiring only some basic calculus, 
for complex models, working out the updates by hand
can be a pain (and often error-prone).

The autograd package expedites this work 
by automatically calculating derivatives. 
And while many other libraries require 
that we compile a symbolic graph to take automatic derivatives, 
`autograd` allows us to take derivatives 
while writing  ordinary imperative code. 
Every time we pass data through our model, 
`autograd` builds a graph on the fly, 
tracking which data combined through 
which operations to produce the output. 
This graph enables `autograd` 
to subsequently backpropagate gradients on command. 
Here, *backpropagate* simply means to trace through the compute graph, 
filling in the partial derivatives with respect to each parameter. 
If you are unfamiliar with some of the math, 
e.g., gradients, please refer to :numref:`chapter_math`.

```{.python .input  n=1}
from mxnet import autograd, np, npx
npx.set_np()
```

## A Simple Example

As a toy example, say that we are interested 
in differentiating the mapping 
$y = 2\mathbf{x}^{\top}\mathbf{x}$ 
with respect to the column vector $\mathbf{x}$. 
To start, let's create the variable `x` and assign it an initial value.

```{.python .input  n=2}
x = np.arange(4)
x
```

Note that before we even calculate the gradient 
of ``y`` with respect to ``x``, 
we will need a place to store it. 
It's important that we do not allocate new memory
every time we take a derivative with respect to a parameter
because we will often update the same parameters 
thousands or millions of times 
and could quickly run out of memory.

Note also that a gradient with respect to a vector $x$ 
is itself vector-valued and has the same shape as $x$.
Thus it is intuitive that in code, 
we will access a gradient taken with respect to `x` 
as an attribute the `ndarray` `x` itself.
We allocate memory for an `ndarray`'s gradient
by invoking its ``attach_grad()`` method.

```{.python .input  n=3}
x.attach_grad()
```

After we calculate a gradient taken with respect to `x`, 
we will be able to access it via the `.grad` attribute. 
As a safe default, `x.grad` initializes as an array containing all zeros.
That's sensible because our most common use case 
for taking gradient in deep learning is to subsequently 
update parameters by adding (or subtracting) the gradient
to maximize (or minimize) the differentiated function.
By initializing the gradient to $\mathbf{0}$,
we ensure that any update accidentally exectuted 
before a gradient has actually been calculated
will not alter the variable's value.

```{.python .input}
x.grad
```

Now let's calculate ``y``. 
Because we wish to subsequently calculate gradients 
we want MXNet to generate a computation graph on the fly. 
We could imagine that MXNet would be turning on a recording device 
to capture the exact path by which each variable is generated.

Note that building the computation graph 
requires a nontrivial amount of computation. 
So MXNet will only build the graph when explicitly told to do so. 
We can invoke this behavior by placing our code 
inside a ``with autograd.record():`` block.

```{.python .input  n=4}
with autograd.record():
    y = 2.0 * np.dot(x, x)
y
```

Since `x` is an `ndarray` of length 4, 
`np.dot` will perform an inner product of `x` and `x`,
yielding the scalar output that we assign to `y`. 
Next, we can automatically calculate the gradient of `y`
with respect to each component of `x` 
by calling `y`'s `backward` function.

```{.python .input  n=5}
y.backward()
```

If we recheck the value of `x.grad`, we will find its contents overwritten by the newly calculated gradient.

```{.python .input}
x.grad
```

The gradient of the function $y = 2\mathbf{x}^{\top}\mathbf{x}$ 
with respect to $\mathbf{x}$ should be $4\mathbf{x}$. 
Let's quickly verify that our desired gradient was calculated correctly.
If the two NDArrays are indeed the same, 
then their difference should consist of all zeros.

```{.python .input  n=6}
x.grad - 4 * x
```

If we subsequently compute the gradient of another variable
whose value was calculated as a function of `x`, 
the contents of `x.grad` will be overwritten.

```{.python .input}
with autograd.record():
    y = x.sum()
y.backward()
x.grad
```

## Backward for Non-scalar Variable

Technically, when `y` is not a scalar, 
the most natural interpretation of the gradient of `y` (a vector of length $m$)
with respect to `x` (a vector of length $n$) is the Jacobian (an $m\times n$ matrix).
For higher-order and higher-dimensional $y$ and $x$, 
the Jacobian could be a gnarly high order tensor 
and complex to compute (refer to :numref:`chapter_math`). 

However, while these more exotic objects do show up 
in advanced machine learning (including in deep learning),
more often when we are calling backward on a vector,
we are trying to calculate the derivatives of the loss functions
for each constitutent of a *batch* of training examples.
Here, our intent is not to calculate the Jacobian
but rather the sum of the partial derivatives 
computed individuall for each example in the batch.

Thus when we invoke backwards on a vector-valued variable,
MXNet assumes that we want the sum of the gradients.
In short, MXNet, will create a new scalar variable 
by summing the elements in `y`,
and compute the gradient of that variable with respect to `x`.

```{.python .input}
with autograd.record():  # y is a vector
    y = x * x
y.backward()

u = x.copy()
u.attach_grad()
with autograd.record():  # v is scalar
    v = (u * u).sum()
v.backward()

x.grad - u.grad
```

## Advanced Autograd

Already you know enough to employ `autograd` and `ndarray` 
successfully to develop many practical models. 
While the rest of this section is not necessary just yet,
we touch on a few advanced topics for completeness. 

### Detach Computations

Sometimes, we wish to  move some calculations 
outside of the recorded computation graph. 
For example, say that `y` was calculated as a function of `x`.
And that subsequently `z` was calcatated a function of both `y` and `x`. 
Now, imagine that we wanted to calculate 
the gradient of `z` with respect to `x`,
but wanted for some reason to treat `y` as a constant,
and only take into account the role 
that `x` played after `y` was calculated.

Here, we can call `u = y.detach()` to return a new variable 
that has the same values as `y` but discards any information
about how `u` was computed. 
In other words, the gradient will not flow backwards through `u` to `x`. 
This will provide the same functionality as if we had
calculated `u` as a function of `x` outside of the `autograd.record` scope, 
yielding a `u` that will be treated as a constant in any called to `backward`. 
The following backward computes $\partial (u \odot x)/\partial x$ 
instead of $\partial (x \odot x \odot x) /\partial x$,
where $\odot$ stands for element-wise multiplication.

```{.python .input}
with autograd.record():
    y = x * x
    u = y.detach()
    z = u * x
z.backward()
x.grad - u
```

Since the computation of $y$ was recorded, 
we can subsequently call `y.backward()` to get $\partial y/\partial x = 2x$.

```{.python .input}
y.backward()
x.grad - 2*x
```

## Attach Gradients to Internal Variables

Attaching gradients to a variable `x` implicitly calls `x=x.detach()`. 
If `x` is computed based on other variables, 
this part of computation will not be used in the backward function.

```{.python .input}
y = np.ones(4) * 2
y.attach_grad()
with autograd.record():
    u = x * y
    u.attach_grad()  # implicitly run u = u.detach()
    z = u + x
z.backward()
print(x.grad, '\n', u.grad, '\n', y.grad)
```

## Head gradients

Detaching allows to breaks the computation into several parts. We could use chain rule :numref:`chapter_math` to compute the gradient for the whole computation.  Assume $u = f(x)$ and $z = g(u)$, by chain rule we have $\frac{dz}{dx} = \frac{dz}{du} \frac{du}{dx}.$ To compute $\frac{dz}{du}$, we can first detach $u$ from the computation and then call `z.backward()` to compute the first term.

```{.python .input}
y = np.ones(4) * 2
y.attach_grad()
with autograd.record():
    u = x * y
    v = u.detach()  # u still keeps the computation graph
    v.attach_grad()
    z = v + x
z.backward()
print(x.grad, '\n', y.grad)
```

Subsequently, we can call `u.backward()` to compute the second term, 
but pass the first term as the head gradients to multiply both terms 
so that `x.grad` will contains $\frac{dz}{dx}$ instead of $\frac{du}{dx}$.

```{.python .input}
u.backward(v.grad)
print(x.grad, '\n', y.grad)
```

## Computing the Gradient of Python Control Flow

One benefit of using automatic differentiation 
is that even if building the computational graph of a function 
required passing through a maze of Python control flow 
(e.g. conditionals, loops, and arbitrary function calls), 
we can still calculate the gradient of the resulting variable. 
In the following snippet, note that 
the number of iterations of the `while` loop 
and the evaluation of the `if` statement
both depend on the value of the input `b`.

```{.python .input  n=8}
def f(a):
    b = a * 2
    while np.abs(b).sum() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

Again to compute gradients, we just need to `record` the calculation
and then call the `backward` function.

```{.python .input  n=9}
a = np.random.normal()
a.attach_grad()
with autograd.record():
    d = f(a)
d.backward()
```

We can now analyze the `f` function defined above. 
Note that it is piecewise linear in its input `a`. 
In other words, for any `a` there exists some constant 
such that for a given range `f(a) = g * a`. 
Consequently `d / a` allows us to verify that the gradient is correct:

```{.python .input  n=10}
print(a.grad == (d / a))
```

## Training Mode and Prediction Mode

As we have seen, after we call `autograd.record`, 
MXNet logs the operations in the following block. 
There is one more subtle detail to be aware of.
Additionally, `autograd.record` will change 
the running mode from *prediction* mode to *training* mode. 
We can verify this behavior by calling the `is_training` function.

```{.python .input  n=7}
print(autograd.is_training())
with autograd.record():
    print(autograd.is_training())
```

When we get to complicated deep learning models,
we will encounter some algorithms where the model
behaves differently during training and 
when we subsequently use it to make predictions. 
The popular neural network techniques *dropout* :numref:`chapter_dropout` 
and *batch normalization* :numref:`chapter_batch_norm`
both exhibit this characteristic.
In other cases, our models may store auxiliary variables in *training* mode 
for purposes of make computing gradients easier 
that are not necessary at prediction time. 
We will cover these differences in detail in later chapters. 


## Summary

* MXNet provides an `autograd` package to automate the calculation of derivatives. To use it, we first attach gradients to those variables with respect to which we desire partial derivartives. We then record the computation of our target value, executed its backward function, and access the resulting gradient via our variable's `grad` attribute.
* We can detach gradients and pass head gradients to the backward function to control the part of the computation will be used in the backward function.
* The running modes of MXNet include *training mode* and *prediction mode*. We can determine the running mode by calling `autograd.is_training()`.

## Exercises

1. Try to run `y.backward()` twice.
1. In the control flow example where we calculate the derivative of `d` with respect to `a`, what would happen if we changed the variable `a` to a random vector or matrix. At this point, the result of the calculation `f(a)` is no longer a scalar. What happens to the result? How do we analyze this?
1. Redesign an example of finding the gradient of the control flow. Run and analyze the result.
1. In a second-price auction (such as in eBay or in computational advertising), the winning bidder pays the second-highest price. Compute the gradient of the final price with respect to the winning bidder's bid using `autograd`. What does the result tell you about the mechanism? If you are curious to learn more about second-price auctions, check out this paper by [Edelman, Ostrovski and Schwartz, 2005](https://www.benedelman.org/publications/gsp-060801.pdf).
1. Why is the second derivative much more expensive to compute than the first derivative?
1. Derive the head gradient relationship for the chain rule. If you get stuck, use the ["Chain rule" article on Wikipedia](https://en.wikipedia.org/wiki/Chain_rule).
1. Assume $f(x) = \sin(x)$. Plot $f(x)$ and $\frac{df(x)}{dx}$ on a graph, where you computed the latter without any symbolic calculations, i.e. without exploiting that $f'(x) = \cos(x)$.

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2318)

![](../img/qr_autograd.svg)
