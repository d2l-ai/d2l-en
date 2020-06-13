# Automatic Differentiation
:label:`sec_autograd`

As we have explained in :numref:`sec_calculus`,
differentiation is a crucial step in nearly all deep learning optimization algorithms.
While the calculations for taking these derivatives are straightforward,
requiring only some basic calculus,
for complex models, working out the updates by hand
can be a pain (and often error-prone).

Deep learning frameworks expedite this work
by automatically calculating derivatives, i.e., *automatic differentiation*.
In practice,
based on our designed model
the system builds a *computational graph*,
tracking which data combined through
which operations to produce the output.
Automatic differentiation enables the system to subsequently backpropagate gradients.
Here, *backpropagate* simply means to trace through the computational graph,
filling in the partial derivatives with respect to each parameter.

```python
from mxnet import autograd, np, npx
npx.set_np()
```


```python
#@tab pytorch
import torch
```


```python
#@tab tensorflow
import tensorflow as tf
```


```{.python .input  n=63}
#@tab jax
import jax.numpy as np
from jax import grad
from jax import random
key = random.PRNGKey(42) #randomness works a bit differently in JAX
```

## A Simple Example

As a toy example, say that we are interested
in differentiating the function
$y = 2\mathbf{x}^{\top}\mathbf{x}$
with respect to the column vector $\mathbf{x}$.
To start, let us create the variable `x` and assign it an initial value.

```python
x = np.arange(4.0)
x
```


```python
#@tab pytorch
x = torch.arange(4.0)
x
```


```python
#@tab tensorflow
x = tf.constant(range(4), dtype=tf.float32)
x
```


```{.python .input  n=2}
#@tab jax
x = np.arange(4., dtype=np.float32)
x
```

```{.json .output n=2}
[
 {
  "data": {
   "text/plain": "DeviceArray([0., 1., 2., 3.], dtype=float32)"
  },
  "execution_count": 2,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Before we even calculate the gradient
of $y$ with respect to $\mathbf{x}$,
we will need a place to store it.
It is important that we do not allocate new memory
every time we take a derivative with respect to a parameter
because we will often update the same parameters
thousands or millions of times
and could quickly run out of memory.
Note that a gradient of a scalar-valued function
with respect to a vector $\mathbf{x}$
is itself vector-valued and has the same shape as $\mathbf{x}$.

```python
# We allocate memory for a tensor's gradient by invoking its `attach_grad`
# method
x.attach_grad()
# After we calculate a gradient taken with respect to `x`, we will be able to
# access it via the `grad` attribute, whose values are initialized with 0s
x.grad
```


```python
#@tab pytorch
x.requires_grad_(True)  # Equals to x = torch.arange(4.0, requires_grad=True)
x.grad  # The default value is None
```


```python
#@tab tensorflow
x = tf.Variable(x)
```


```{.python .input  n=3}
#@tab jax
# Jax doesn't store gradients in objects but has a functional API instead
```

Now let us calculate $y$.

```python
# Place our code inside an `autograd.record` scope to build the computational
# graph
with autograd.record():
    y = 2 * np.dot(x, x)
y
```


```python
#@tab pytorch
y = 2 * torch.dot(x, x)
y
```


```python
#@tab tensorflow
# Record all computations onto a tape. 
with tf.GradientTape() as t:
    y = 2 * tf.tensordot(x, x, axes=1)
y
```


```{.python .input  n=9}
#@tab jax
f = lambda x: 2*np.dot(x,x)
y = f(x)
y
```

```{.json .output n=9}
[
 {
  "data": {
   "text/plain": "DeviceArray(28., dtype=float32)"
  },
  "execution_count": 9,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Since `x` is a vector of length 4,
an inner product of `x` and `x` is performed,
yielding the scalar output that we assign to `y`.
Next, we can automatically calculate the gradient of `y`
with respect to each component of `x`
by calling the function for backpropagation and printing the gradient.

```python
y.backward()
x.grad
```


```python
#@tab pytorch
y.backward()
x.grad
```


```python
#@tab tensorflow
x_grad = t.gradient(y, x)
x_grad
```


```{.python .input  n=10}
#@tab jax
#Take the gradient of y w.r.t. the first argument
x_grad = grad(f)(x)
x_grad
```

```{.json .output n=10}
[
 {
  "data": {
   "text/plain": "DeviceArray([ 0.,  4.,  8., 12.], dtype=float32)"
  },
  "execution_count": 10,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

The gradient of the function $y = 2\mathbf{x}^{\top}\mathbf{x}$
with respect to $\mathbf{x}$ should be $4\mathbf{x}$.
Let us quickly verify that our desired gradient was calculated correctly.

```python
x.grad == 4 * x
```


```python
#@tab pytorch
x.grad == 4 * x
```


```python
#@tab tensorflow
x_grad == 4 * x
```


```{.python .input  n=18}
#@tab jax
x_grad == 4 * x
```

```{.json .output n=18}
[
 {
  "data": {
   "text/plain": "DeviceArray([ True,  True,  True,  True], dtype=bool)"
  },
  "execution_count": 18,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Now let us calculate another function of `x`.

```python
with autograd.record():
    y = x.sum()
y.backward()
x.grad  # Overwritten by the newly calculated gradient.
```


```python
#@tab pytorch
# PyTorch accumulates the gradient in default, we need to clear the previous 
# values.
x.grad.zero_() 
y = x.sum()
y.backward()
x.grad
```


```python
#@tab tensorflow
with tf.GradientTape() as t:
    y = tf.reduce_sum(x)
t.gradient(y, x)  # Overwritten by the newly calculated gradient.
```


```{.python .input  n=20}
#@tab jax
f = lambda x: np.sum(x)

y = f(x)
grad(f)(x)
```

```{.json .output n=20}
[
 {
  "data": {
   "text/plain": "DeviceArray([1., 1., 1., 1.], dtype=float32)"
  },
  "execution_count": 20,
  "metadata": {},
  "output_type": "execute_result"
 }
]
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

```python
# When we invoke `backward` on a vector-valued variable `y` (function of `x`),
# a new scalar variable is created by summing the elements in `y`. Then the
# gradient of that scalar variable with respect to `x` is computed
with autograd.record():
    y = x * x  # `y` is a vector
y.backward()
x.grad  # Equals to y = sum(x * x)
```


```python
#@tab pytorch
x.grad.zero_()
y = x * x
y.sum().backward()  # Backward only supports for scalars. 
x.grad
```


```python
#@tab tensorflow
with tf.GradientTape() as t:
    y = x * x
t.gradient(y, x)  # Equals to y = tf.reduce_sum(x * x)
```


JAX only supports scalar-value functions, hence for non-scalar variables we have to perform a reductive summation.

```{.python .input  n=21}
#@tab jax
f = lambda x: np.sum(x*x)

y = f(x)
grad(f)(x)
```

```{.json .output n=21}
[
 {
  "data": {
   "text/plain": "DeviceArray([0., 2., 4., 6.], dtype=float32)"
  },
  "execution_count": 21,
  "metadata": {},
  "output_type": "execute_result"
 }
]
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

Here, we can detach `y` to return a new variable `u`
that has the same value as `y` but discards any information
about how `y` was computed in the computational graph.
In other words, the gradient will not flow backwards through `u` to `x`.
This will provide the same functionality as if we had
calculated `u` as a function of `x` outside of the scope,
yielding a `u` that will be treated as a constant in any backward call.
Thus, the following backward function computes
the partial derivative of `z = u * x` with respect to `x` while treating `u` as a constant,
instead of the partial derivative of `z = x * x * x` with respect to `x`.

```python
with autograd.record():
    y = x * x
    u = y.detach()
    z = u * x
z.backward()
x.grad == u
```


```python
#@tab pytorch
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u
```


```python
#@tab tensorflow
# Set the persistent=True to run t.gradient more than once.
with tf.GradientTape(persistent=True) as t:
    y = x * x
    u = tf.stop_gradient(y)
    z = u * x

x_grad = t.gradient(z, x)
x_grad == u
```


JAX does not require any detachment of computation since it relies on side-effect-free functions which don't mutate global state. You have to explicitly define which function to differentiate and with respect to which argument.

```{.python .input  n=37}
y = (lambda a: a*2)(x)
```

```{.python .input  n=53}
z = (lambda a: np.sum(y*a))
```

```{.python .input  n=56}
grad(z)(x)
```

```{.json .output n=56}
[
 {
  "data": {
   "text/plain": "DeviceArray([0., 2., 4., 6.], dtype=float32)"
  },
  "execution_count": 56,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=59}
#@tab jax
y = (lambda a: a*2)(x)

z = (lambda a: np.sum(y*a))

x_grad = grad(z)(x)
x_grad
```

```{.json .output n=59}
[
 {
  "data": {
   "text/plain": "DeviceArray([0., 2., 4., 6.], dtype=float32)"
  },
  "execution_count": 59,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Since the computation of `y` was recorded,
we can subsequently call backward function on `y` to get the derivative of `y = x * x` with respect to `x`, which is `2 * x`.

```python
y.backward()
x.grad == 2 * x
```


```python
#@tab pytorch
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x
```


```python
#@tab tensorflow
t.gradient(y, x) == 2 * x
```


```{.python .input  n=60}
#@tab jax
x_grad == 2 * x
```

```{.json .output n=60}
[
 {
  "data": {
   "text/plain": "DeviceArray([ True,  True,  True,  True], dtype=bool)"
  },
  "execution_count": 60,
  "metadata": {},
  "output_type": "execute_result"
 }
]
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

```python
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


```python
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


```python
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


```{.python .input  n=61}
#@tab jax
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

Let us compute the gradient.

```python
a = np.random.normal()
a.attach_grad()
with autograd.record():
    d = f(a)
d.backward()
```


```python
#@tab pytorch
a = torch.randn(size=(1,), requires_grad=True)
d = f(a)
d.backward()
```


```python
#@tab tensorflow
a = tf.Variable(tf.random.normal((1, 1),dtype=tf.float32))
with tf.GradientTape() as t:
    d = f(a)
d_grad = t.gradient(d, a)
d_grad
```


```{.python .input  n=68}
#@tab jax
a = random.normal(key, dtype=np.float32) #JAX requires a random key
d = f(a)
d_grad = grad(f)(a)
d_grad
```

```{.json .output n=68}
[
 {
  "data": {
   "text/plain": "DeviceArray(819200., dtype=float32)"
  },
  "execution_count": 68,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

We can now analyze the `f` function defined above.
Note that it is piecewise linear in its input `a`.
In other words, for any `a` there exists some constant scalar `k`
such that `f(a) = k * a`, where the value of `k` depends on the input `a`.
Consequently `d / a` allows us to verify that the gradient is correct.

```python
a.grad == d / a
```


```python
#@tab pytorch
a.grad == (d / a)
```


```python
#@tab tensorflow
d_grad == (d / a)
```


```{.python .input  n=69}
#@tab jax
d_grad == d / a
```

```{.json .output n=69}
[
 {
  "data": {
   "text/plain": "DeviceArray(True, dtype=bool)"
  },
  "execution_count": 69,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## Summary

* Deep learning frameworks can automate the calculation of derivatives. To use it, we first attach gradients to those variables with respect to which we desire partial derivatives. We then record the computation of our target value, execute its function for backpropagation, and access the resulting gradient.


## Exercises

1. Why is the second derivative much more expensive to compute than the first derivative?
1. After running the function for backpropagation, immediately run it again and see what happens.
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
