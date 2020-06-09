# Automatic Differentiation
:label:`sec_autograd`

As we have explained in :numref:`sec_calculus`,
differentiation is a crucial step in nearly all deep learning optimization algorithms.
While the calculations for taking these derivatives are straightforward,
requiring only some basic calculus,
for complex models, working out the updates by hand
can be a pain (and often error-prone).


:begin_tab:`mxnet`
The `autograd` package expedites this work 
by automatically calculating derivatives, i.e., *automatic differentiation*. 
And while many other libraries require 
that we compile a symbolic graph to take automatic derivatives, 
`autograd` allows us to take derivatives 
while writing ordinary imperative code. 
Every time we pass data through our model, 
`autograd` builds a graph on the fly, 
tracking which data combined through 
which operations to produce the output. 
This graph enables `autograd` 
to subsequently backpropagate gradients on command. 
Here, *backpropagate* simply means to trace through the *computational graph*,
filling in the partial derivatives with respect to each parameter.
:end_tab:

:begin_tab:`pytorch`
Deep learning frameworks can expedite this work
by automatically calculating derivatives, i.e., *automatic differentiation*.
And while many other libraries require
that we compile a symbolic graph to take automatic derivatives,
PyTorch allows us to take derivatives
while writing ordinary imperative code.
Every time we pass data through our model,
they build a graph on the fly,
tracking which data combined through
which operations to produce the output.
This graph enables PyTorch
to subsequently backpropagate gradients on command.
Here, *backpropagate* simply means to trace through the *computational graph*,
filling in the partial derivatives with respect to each parameter.
:end_tab:

```{.python .input}
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
import torch
```

## A Simple Example

As a toy example, say that we are interested
in differentiating the function
$y = 2\mathbf{x}^{\top}\mathbf{x}$
with respect to the column vector $\mathbf{x}$.
To start, let us create the variable `x` and assign it an initial value.

```{.python .input}
x = np.arange(4.0)
x
```

```{.python .input}
#@tab pytorch
x = torch.arange(4.0, requires_grad=True)
x
```

Note that before we even calculate the gradient
of $y$ with respect to $\mathbf{x}$,
we will need a place to store it.
It is important that we do not allocate new memory
every time we take a derivative with respect to a parameter
because we will often update the same parameters
thousands or millions of times
and could quickly run out of memory.

Note also that a gradient of a scalar-valued function
with respect to a vector $\mathbf{x}$
is itself vector-valued and has the same shape as $\mathbf{x}$.
Thus it is intuitive that in code,
we will access a gradient taken with respect to `x`
as an attribute of the tensor `x` itself.

:begin_tab:`mxnet`

We allocate memory for a tensor's gradient
by invoking its `attach_grad` method.
After we calculate a gradient taken with respect to `x`,
we will be able to access it via the `grad` attribute.
As a safe default, `x.grad` is initialized as an array containing all zeros.
That is sensible because our most common use case
for taking gradient in deep learning is to subsequently
update parameters by adding (or subtracting) the gradient
to maximize (or minimize) the differentiated function.
By initializing the gradient to an array of zeros,
we ensure that any update accidentally executed
before a gradient has actually been calculated
will not alter the parameters' value.

:end_tab:

:begin_tab:`pytorch`

Note the `requires_grad=True` argument when creating `x`, it tells the framework
we need allocate gradient space for `x` in the future.

:end_tab:

```{.python .input}
x.attach_grad()
x.grad
```

```{.python .input}
#@tab pytorch
x.grad
```

:begin_tab:`mxnet`
Now let us calculate $y$.
Because we wish to subsequently calculate gradients,
we want MXNet to generate a computational graph on the fly.
We could imagine that MXNet would be turning on a recording device
to capture the exact path by which each variable is generated.

Note that building the computational graph
requires a nontrivial amount of computation.
So MXNet will only build the graph when explicitly told to do so.
We can invoke this behavior by placing our code
inside an `autograd.record` scope.
:end_tab:

:begin_tab:`pytorch`
Now let us calculate $y$.
:end_tab:

```{.python .input}
with autograd.record():
    y = 2 * np.dot(x, x)
y
```

```{.python .input}
#@tab pytorch
y = 2 * torch.dot(x, x)
y
```

Since `x` is a tensor of length 4,
the `dot` operator will perform an inner product of `x` and `x`,
yielding the scalar output that we assign to `y`.
Next, we can automatically calculate the gradient of `y`
with respect to each component of `x`
by calling `y`'s `backward` function.

```{.python .input}
y.backward()
```

```{.python .input}
#@tab pytorch
y.backward()
```

If we recheck the value of `x.grad`, we will find its contents overwritten by the newly calculated gradient.

```{.python .input}
x.grad
```

```{.python .input}
#@tab pytorch
x.grad
```

The gradient of the function $y = 2\mathbf{x}^{\top}\mathbf{x}$
with respect to $\mathbf{x}$ should be $4\mathbf{x}$.
Let us quickly verify that our desired gradient was calculated correctly.
If the two tensors are indeed the same,
then the equality between them holds at every position.

```{.python .input}
x.grad == 4 * x
```

```{.python .input}
#@tab pytorch
x.grad == 4 * x
```

:begin_tab:`mxnet`

If we subsequently compute the gradient of another variable
whose value was calculated as a function of `x`,
the contents of `x.grad` will be overwritten.

:end_tab:

:begin_tab:`pytorch`

If we subsequently compute the gradient of another variable
whose value was calculated as a function of `x`, we need to clear the previous
values in `x.grad` first, as PyTorch accumulates and adds the gradient in default.

:end_tab:

```{.python .input}
with autograd.record():
    y = x.sum()
y.backward()
x.grad
```

```{.python .input}
#@tab pytorch
x.grad.zero_()
y = x.sum()
y.backward()
x.grad
```

:begin_tab:`mxnet`

## Backward for Non-Scalar Variables

Technically, when `y` is not a scalar,
the most natural interpretation of the differentiation of a vector `y`
with respect to a vector `x` is a matrix.
For higher-order and higher-dimensional `y` and `x`,
the differentiation result could be a gnarly high-order tensor.

However, while these more exotic objects do show up
in advanced machine learning (including in deep learning),
more often when we are calling backward on a vector,
we are trying to calculate the derivatives of the loss functions
for each constituent of a *batch* of training examples.
Here, our intent is not to calculate the differentiation matrix
but rather the sum of the partial derivatives
computed individually for each example in the batch.

Thus when we invoke `backward` on a vector-valued variable `y`,
which is a function of `x`,
MXNet assumes that we want the sum of the gradients.
In short, MXNet will create a new scalar variable
by summing the elements in `y`,
and compute the gradient of that scalar variable with respect to `x`.

:end_tab:

```{.python .input}
with autograd.record():
    y = x * x  # `y` is a vector
y.backward()

u = x.copy()
u.attach_grad()
with autograd.record():
    v = (u * u).sum()  # `v` is a scalar
v.backward()

x.grad == u.grad
```

## Detaching Computation

Sometimes, we wish to move some calculations
outside of the recorded computational graph.
For example, say that `y` was calculated as a function of `x`,
and that subsequently `z` was calculated as a function of both `y` and `x`.
Now, imagine that we wanted to calculate
the gradient of `z` with respect to `x`,
but wanted for some reason to treat `y` as a constant,
and only take into account the role
that `x` played after `y` was calculated.

Here, we can call `u = y.detach()` to return a new variable `u`
that has the same value as `y` but discards any information
about how `y` was computed in the computational graph.
In other words, the gradient will not flow backwards through `u` to `x`.
This will provide the same functionality as if we had
calculated `u` as a function of `x` outside of the `autograd.record` scope,
yielding a `u` that will be treated as a constant in any `backward` call.
Thus, the following `backward` function computes
the partial derivative of `z = u * x` with respect to `x` while treating `u` as a constant,
instead of the partial derivative of `z = x * x * x` with respect to `x`.

```{.python .input}
with autograd.record():
    y = x * x
    u = y.detach()
    z = u * x
z.backward()
x.grad == u
```

```{.python .input}
#@tab pytorch
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u
```

Since the computation of `y` was recorded,
we can subsequently call `y.backward()` to get the derivative of `y = x * x` with respect to `x`, which is `2 * x`.

```{.python .input}
y.backward()
x.grad == 2 * x
```

```{.python .input}
#@tab pytorch
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x
```

## Computing the Gradient of Python Control Flow

One benefit of using automatic differentiation
is that even if building the computational graph of a function
required passing through a maze of Python control flow
(e.g., conditionals, loops, and arbitrary function calls),
we can still calculate the gradient of the resulting variable.
In the following snippet, note that
the number of iterations of the `while` loop
and the evaluation of the `if` statement
both depend on the value of the input `a`.

```{.python .input}
def f(a):
    b = a * 2
    while np.linalg.norm(b) < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
#@tab pytorch
def f(a):
    b = a * 2
    while b.norm().item() < 1000:
        b = b * 2
    if b.sum().item() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

Again to compute gradients, we just need to `record` the calculation
and then call the `backward` function.

```{.python .input}
a = np.random.normal()
a.attach_grad()
with autograd.record():
    d = f(a)
d.backward()
```

```{.python .input}
#@tab pytorch
a = torch.randn(size=(1,), requires_grad=True)
d = f(a)
d.backward()
```

We can now analyze the `f` function defined above.
Note that it is piecewise linear in its input `a`.
In other words, for any `a` there exists some constant scalar `k`
such that `f(a) = k * a`, where the value of `k` depends on the input `a`.
Consequently `d / a` allows us to verify that the gradient is correct.

```{.python .input}
a.grad == d / a
```

```{.python .input}
#@tab pytorch
a.grad == (d / a)
```

## Training Mode and Prediction Mode

When we get to complicated deep learning models,
we will encounter some algorithms where the model
behaves differently during training and
when we subsequently use it to make predictions.

:begin_tab:`mxnet`

The running mode can be either *training mode* or *prediction mode*.
This can be told when the `is_training` function is called,
which indicates if we are in the `autograd.record` scope or not.

:end_tab:

:begin_tab:`pytorch`

In PyTorch, we can set `model.train()` or `model.eval()` to distinguish the
training mode and prediction mode, which we'll cover in the later sections of
the book.

:end_tab:

```{.python .input}
print(autograd.is_training())
with autograd.record():
    print(autograd.is_training())
```

## Summary

* Deep learning frameworks can automate the calculation of derivatives. To use it, we first attach gradients to those variables with respect to which we desire partial derivatives. We then record the computation of our target value, execute its `backward` function, and access the resulting gradient via our variable's `grad` attribute.
* We can detach gradients to control the part of the computation that will be used in the `backward` function.
* The running modes include training mode and prediction mode.


## Exercises

1. Why is the second derivative much more expensive to compute than the first derivative?
1. After running `y.backward()`, immediately run it again and see what happens.
1. In the control flow example where we calculate the derivative of `d` with respect to `a`, what would happen if we changed the variable `a` to a random vector or matrix. At this point, the result of the calculation `f(a)` is no longer a scalar. What happens to the result? How do we analyze this?
1. Redesign an example of finding the gradient of the control flow. Run and analyze the result.
1. Let $f(x) = \sin(x)$. Plot $f(x)$ and $\frac{df(x)}{dx}$, where the latter is computed without exploiting that $f'(x) = \cos(x)$.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/34)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/35)
:end_tab:
