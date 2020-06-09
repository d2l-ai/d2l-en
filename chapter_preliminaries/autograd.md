# Automatic Differentiation
:label:`sec_autograd`

As we have explained in :numref:`sec_calculus`,
differentiation is a crucial step in nearly all deep learning optimization algorithms.
While the calculations for taking these derivatives are straightforward,
requiring only some basic calculus,
for complex models, working out the updates by hand
can be a pain (and often error-prone).

Deep learning frameworks can expedite this work
by automatically calculating derivatives, i.e., [*automatic differentiation*](https://en.wikipedia.org/wiki/Automatic_differentiation), which is also called *AD*, *autodiff* or *autograd*. There are two distinct modes in autograd, forward mode and reverse mode. Deep learning frameworks in default use the reverse mode, which is straightforward to implement the [backpropagation](https://en.wikipedia.org/wiki/Backpropagation) for training neural networks.

```{.python .input}
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
import torch
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf
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
x = torch.arange(4.0)
x
```

```{.python .input}
#@tab tensorflow
x = tf.constant(range(4), dtype=tf.float32)
x
```

Before we even calculate the gradient
of $y$ with respect to $\mathbf{x}$,
we will need a place to store it.
It is important that we do not allocate new memory
every time we take a derivative with respect to a parameter
because we will often update the same parameters
thousands or millions of times
and could quickly run out of memory.

```{.python .input}
x.attach_grad()
x.grad  # Values are initialized with 0s.
```

```{.python .input}
#@tab pytorch
x.requires_grad_(True)  # Equals to x = torch.arange(4.0, requires_grad=True)
x.grad  # The default value is None
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    t.watch(x)
```

Now let us calculate $y$.

```{.python .input}
# Record all computations in the autograd scope.
with autograd.record():
    y = 2 * np.dot(x, x)
y
```

```{.python .input}
#@tab pytorch
y = 2 * torch.dot(x, x)
y
```

```{.python .input}
#@tab tensorflow
# Record all computations onto a tape. 
with tf.GradientTape() as t:
    t.watch(x)
    y = 2 * tf.tensordot(x, x, axes=1)
y
```

Since `x` is a vector of length 4,
the dot operator will perform an inner product of `x` and `x`,
yielding the scalar output that we assign to `y`.
Next, we can automatically calculate the gradient of `y`
with respect to each component of `x`
by calling the backward function and print the gradient.

```{.python .input}
y.backward()
x.grad
```

```{.python .input}
#@tab pytorch
y.backward()
x.grad
```

```{.python .input}
#@tab tensorflow
x_grad = t.gradient(y, x)
x_grad
```

The gradient of the function $y = 2\mathbf{x}^{\top}\mathbf{x}$
with respect to $\mathbf{x}$ should be $4\mathbf{x}$.
Let us quickly verify that our desired gradient was calculated correctly.

```{.python .input}
x.grad == 4 * x
```

```{.python .input}
#@tab pytorch
x.grad == 4 * x
```

```{.python .input}
#@tab tensorflow
x_grad == 4 * x
```

Let's calculate another function of `x`.

```{.python .input}
with autograd.record():
    y = x.sum()
y.backward()
x.grad  # Overwritten by the newly calculated gradient.
```

```{.python .input}
#@tab pytorch
# PyTorch accumulates the gradient in default, we need to clear the previous 
# values.
x.grad.zero_() 
y = x.sum()
y.backward()
x.grad
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    t.watch(x)
    y = tf.reduce_sum(x)
t.gradient(y, x)  # Overwritten by the newly calculated gradient.
```

## Backward for Non-Scalar Variables

Technically, when `y` is not a scalar,
the most natural interpretation of the differentiation of a vector `y`
with respect to a vector `x` is a matrix.
For higher-order and higher-dimensional `y` and `x`,
the differentiation result could be a high-order tensor.

However, while these more exotic objects do show up
in advanced machine learning (including in deep learning),
more often when we are calling backward on a vector,
we are trying to calculate the derivatives of the loss functions
for each constituent of a *batch* of training examples.
Here, our intent is not to calculate the differentiation matrix
but rather the sum of the partial derivatives
computed individually for each example in the batch.

```{.python .input}
with autograd.record():
    y = x * x  # `y` is a vector
y.backward()  
x.grad  # Equals to y = sum(x*x)
```

```{.python .input}
#@tab pytorch
x.grad.zero_()
y = x * x
y.sum().backward()  # Backward only supports for scalars. 
x.grad
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    t.watch(x)
    y = x * x
t.gradient(y, x)  # Equals to y = tf.reduce_sum(x*x)
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

One benefit of using autograd
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

```{.python .input}
#@tab tensorflow
def f(a):
    b = a * 2
    while tf.norm(b) < 1000:
        b = b * 2
    if tf.reduce_sum(b) > 0:
        c = b
    else:
        c = 100 * b
    return c
```

Let's compute the gradient.

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

```{.python .input}
#@tab tensorflow
a = tf.random.normal((1, 1),dtype=tf.float32)
with tf.GradientTape() as t:
    t.watch(a)
    d = f(a)
d_grad = t.gradient(d, a)
d_grad
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

```{.python .input}
#@tab tensorflow
d_grad == (d / a)
```

## Summary

* Deep learning frameworks can automate the calculation of derivatives. To use it, we first attach gradients to those variables with respect to which we desire partial derivatives. We then record the computation of our target value, execute its backward function, and access the resulting gradient.


## Exercises

1. Why is the second derivative much more expensive to compute than the first derivative?
1. After running the backward function, immediately run it again and see what happens.
1. In the control flow example where we calculate the derivative of `d` with respect to `a`, what would happen if we changed the variable `a` to a random vector or matrix. At this point, the result of the calculation `f(a)` is no longer a scalar. What happens to the result? How do we analyze this?
1. Redesign an example of finding the gradient of the control flow. Run and analyze the result.
1. Let $f(x) = \sin(x)$. Plot $f(x)$ and $\frac{df(x)}{dx}$, where the latter is computed without exploiting that $f'(x) = \cos(x)$.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/34)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/35)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/200)
:end_tab:
