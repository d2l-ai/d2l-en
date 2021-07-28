# Automatic Differentiation
:label:`sec_autograd`

As explained in :numref:`sec_calculus`,
differentiation is a crucial step in nearly all deep learning optimization algorithms.
While the calculations for taking these derivatives are straightforward,
working out the updates by hand can tedious and error-prone, particularly for complex models.

Deep learning frameworks expedite this work via *automatic differentiation*. 
Based on our designed model the system builds a *computational graph*. 
This allows us to track which data is needed for a given output. 
Automatic differentiation enables the system to backpropagate gradients.
Here, *backpropagation* simply means to trace through the computational graph,
filling in the partial derivatives with respect to each parameter. 

While automatic differentiation, or in short, autograd, has gained significant 
prominence over the past decade, it has a long history. One of the first references is the work by 
:cite:`wengert1964simple` who introduced automatic differentiation over half a century ago. The
description of backpropagation of the form that is used in deep learning systems did not 
happen until the PhD thesis of :cite:`speelpenning1980compiling`. More modern descriptions 
can be found in the work of :cite:`griewank1989automatic` who put autograd on a comprehensive footing. 
While by now backpropagation is the default choice in computing gradients, it is by far not the only option. For instance Julia employs forward propagation :cite:`revels2016forward`. We will discuss this in some more detail after we've introduced the basics.


## A Simple Function

Let's assume that we are interested
in (**differentiating the function
$y = 2\mathbf{x}^{\top}\mathbf{x}$
with respect to the column vector $\mathbf{x}$.**) To start, let us create the variable `x` and assign it an initial value.

```{.python .input}
from mxnet import autograd, np, npx
npx.set_np()

x = np.arange(4.0)
x
```

```{.python .input  n=3}
#@tab pytorch
import torch

x = torch.arange(4.0)
x
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

x = tf.range(4, dtype=tf.float32)
x
```

[**Before we even calculate the gradient
of $y$ with respect to $\mathbf{x}$,
we will need a place to store it.**]
It is important that we do not allocate new memory
every time we take a derivative with respect to a parameter
because we will often update the same parameters
thousands or millions of times
and could quickly run out of memory.
Note that a gradient of a scalar-valued function
with respect to a vector $\mathbf{x}$
is itself vector-valued and has the same shape as $\mathbf{x}$.

```{.python .input}
# We allocate memory for a tensor's gradient by invoking `attach_grad`
x.attach_grad()
# After we calculate a gradient taken with respect to `x`, we will be able to
# access it via the `grad` attribute, whose values are initialized with 0s
x.grad
```

```{.python .input  n=4}
#@tab pytorch
x.requires_grad_(True)  # Better create `x = torch.arange(4.0, requires_grad=True)`
x.grad                  # The default value is None
```

```{.python .input}
#@tab tensorflow
x = tf.Variable(x)
```

(**Now let us calculate $y$.**)

```{.python .input}
# Our code is inside an `autograd.record` scope to build the computational graph
with autograd.record():
    y = 2 * np.dot(x, x)
y
```

```{.python .input  n=5}
#@tab pytorch
y = 2 * torch.dot(x, x)
y
```

```{.python .input}
#@tab tensorflow
# Record all computations onto a tape
with tf.GradientTape() as t:
    y = 2 * tf.tensordot(x, x, axes=1)
y
```

While `x` is a vector of length 4, the dot product yields a scalar 
output that we assign to `y`. 
Next, [**we can automatically calculate the gradient of `y`
with respect to each component of `x`**]
by calling the function for backpropagation and printing the gradient.

```{.python .input}
y.backward()
x.grad
```

```{.python .input  n=6}
#@tab pytorch
y.backward()
x.grad
```

```{.python .input}
#@tab tensorflow
x_grad = t.gradient(y, x)
x_grad
```

(**The gradient of the function $y = 2\mathbf{x}^{\top}\mathbf{x}$
with respect to $\mathbf{x}$ should be $4\mathbf{x}$.**)
Let us verify that the gradient was calculated correctly. Since the automatic gradient computation was done symbolically we have every reason to expect that the two values are identical as opposed to being close within some level of numerical precision.

```{.python .input}
x.grad == 4 * x
```

```{.python .input  n=7}
#@tab pytorch
x.grad == 4 * x
```

```{.python .input}
#@tab tensorflow
x_grad == 4 * x
```

:begin_tab:`mxnet`
[**Let us calculate another function of `x`.**] MXNet resets the gradient buffer whenever we record a new gradient. 
:end_tab:

:begin_tab:`pytorch`
[**Let us calculate another function of `x`.**] PyTorch does not reset the gradient buffer whenever we record a new gradient. This can be convenient when we have multiple objective functions within a single optimization problem and where we want to aggregate all gradients into one term. In the current context we need to clear the gradient explicitly to get a correct result for the new function.
:end_tab:

:begin_tab:`tensorflow`
[**Let us calculate another function of `x`.**] TensorFlow resets the gradient buffer whenever we record a new gradient. 
:end_tab:

```{.python .input}
with autograd.record():
    y = x.sum()
y.backward()
x.grad  # Overwritten by the newly calculated gradient
```

```{.python .input  n=8}
#@tab pytorch
x.grad.zero_()  # Reset gradient
y = x.sum()
y.backward()
x.grad
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = tf.reduce_sum(x)
t.gradient(y, x)  # Overwritten by the newly calculated gradient
```

## Backward for Non-Scalar Variables

When `y` is a vector,
the most natural interpretation of the derivative of  `y`
with respect to a vector `x` is a matrix. Likewise, for 
higher-order and higher-dimensional `y` and `x`,
the differentiation result could be an even higher-order tensor.

While these more exotic objects do show up
in advanced machine learning (including [**in deep learning**]),
more often than not (**when we are calling backward on a vector**) 
our goal is more modest: we are trying to calculate the derivatives of the loss functions
for each constituent of a *batch* of training examples.
Here, (**our intent is**) not to calculate the matrix (it is called the Jacobian)
but rather (**the sum of the partial derivatives
computed individually for each example**) in the batch.

:begin_tab:`mxnet`
MXNet handles this problem by reducing all tensors to scalars by summing before computing a gradient. 
This way we can keep the gradient size at bay. In other words, rather than returning the Jacobian $\partial_{\mathbf{x}} \mathbf{y}$ it returns the gradient of $\partial_{\mathbf{x}} \sum_i y_i$ instead. 
:end_tab:

:begin_tab:`pytorch`
Invoking `backward` on a non-scalar variable results in an error unless we tell PyTorch how to reduce the object to a scalar. More formally, we need to provide some vector $\mathbf{v}$ such that the gradient computation can generate $\mathbf{v}^\top \partial_{\mathbf{x}} \mathbf{y}$ rather than $\partial_{\mathbf{x}} \mathbf{y}$. For reasons that will become clear later when we construct large computational graphs, this argument is referred to as `gradient`. For a much more detailed description see Yang Zhang's [Medium post](https://zhang-yang.medium.com/the-gradient-argument-in-pytorchs-backward-function-explained-by-examples-68f266950c29). 
:end_tab:

:begin_tab:`tensorflow`
TensorFlow resets the gradient buffer whenever we record a new gradient. Moreover, it aggregates by default all coordinates of the tensor. In other words, rather than returning the Jacobian $\partial_{\mathbf{x}} \mathbf{y}$ it returns the gradient of $\partial_{\mathbf{x}} \sum_i y_i$ instead. 
:end_tab:

```{.python .input}
with autograd.record():
    y = x * x  
y.backward()
x.grad  # Equals the gradient of y = sum(x * x)
```

```{.python .input}
#@tab pytorch
x.grad.zero_()
y = x * x
y.backward(gradient=torch.ones(len(y)))  # Faster: y.sum().backward()
x.grad
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = x * x
t.gradient(y, x)  # Same as `y = tf.reduce_sum(x * x)`
```

## Detaching Computation

Sometimes, we wish to [**move some calculations
outside of the recorded computational graph.**]
For example, say that we use the input to create some auxiliary intermediate terms for which we do not want to compute a gradient. In this case we need to *detach* the respective computational influence graph from the final result. The following toy example will make this more obvious: suppose we have `z = x * y` and `y = x * x` but we want to focus on the *direct* influence of `x` on `z` rather than the influence conveyed via `y`. In this case we can create a new variable `u = y` whose provenance has been detached from `x`. Thus backpropagation will return `x` as the gradient of `z = x * u` instead of `3 * x * x`, as the gradient of `z = x * x * x` would imply.

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

```{.python .input}
#@tab tensorflow
# Set `persistent=True` to preserve the compute graph. 
# This lets us run `t.gradient` more than once
with tf.GradientTape(persistent=True) as t:
    y = x * x
    u = tf.stop_gradient(y)
    z = u * x

x_grad = t.gradient(z, x)
x_grad == u
```

Note that even though we detached `y` from the graph leading to `z`, the graph leading to `y` has still been recorded and we can retrieve the gradient of `y` with respect to `x` subsequently. Let's look at how this works in practice.

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

```{.python .input}
#@tab tensorflow
t.gradient(y, x) == 2 * x
```

## Gradients and Python Control Flow

So far we reviewed cases where the path from input to result was well-defined via a function such as $z = x^3$.  Programming offers us a lot more freedom in how we compute results. For instance, we can make them depend on auxiliary variables or condition choices on intermediate results. One benefit of using automatic differentiation
is that [**even if**] building the computational graph of (**a function
required passing through a maze of Python control flow**)
(e.g., conditionals, loops, and arbitrary function calls),
(**we can still calculate the gradient of the resulting variable.**)
To illustrate this, consider the following code snippet where 
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
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
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

Let us compute the gradient. Since the input is a random variable, we cannot predict the flow a priori. It is only by executing `f(a)` that we know which operations will be carried out, and how frequently. 

```{.python .input}
a = np.random.normal()
a.attach_grad()
with autograd.record():
    d = f(a)
d.backward()
```

```{.python .input}
#@tab pytorch
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
```

```{.python .input}
#@tab tensorflow
a = tf.Variable(tf.random.normal(shape=()))
with tf.GradientTape() as t:
    d = f(a)
d_grad = t.gradient(d, a)
d_grad
```

Even though `f(a)` seems to be a rather convoluted function that we made appear overly complex for demonstration purposes, its dependence on the input is actually quite simple: it is a *linear* function of `a` with piecewise defined scale. As such, `f(a) / a` is a vector of constant entries and moreover, `f(a) / a` needs to match the gradient of `f(a)` with regard to `a`. Let's verify this:

```{.python .input}
a.grad == d / a
```

```{.python .input}
#@tab pytorch
a.grad == d / a
```

```{.python .input}
#@tab tensorflow
d_grad == d / a
```

Dynamic control flow is very common in Deep Learning. For instance, when processing text, the output will depend on the length of the input. Likewise, when detecting objects in images, the number of detections should depend on the number of objects that can be found (e.g., counting how many sheep can be seen in a photo of a pasture). This is one of the key aspects where automatic differentiation becomes vital for statistical modeling since it is impossible to compute the gradient manually a priori. 


## Summary

In this section we got a first taste of the power of automatic differentiation. It is one of the key productivity tools for deep learning modeling as it liberates the user to focus on the desired output of the model rather than the path how the output was obtained. This allows us to design much larger and more complex models than what we would be able to handle with pen and paper when computing derivatives manually. As we will see later, automatic differentiation also makes computing gradients a matter of automation and of *optimization*. That is, framework designers can use tools from compilers and graph manipulation to compute the results in the most expedient and memory-efficient manner. 

For now, it is sufficient for us to remember the basics: attach gradients to those variables with respect to which we desire partial derivatives. We then record the computation of our target value, execute its function for backpropagation, and access the resulting gradient. The remainder of the book will largely follow this pattern.


## Exercises

1. Why is the second derivative much more expensive to compute than the first derivative?
1. After running the function for backpropagation, immediately run it again and see what happens. Why?
1. In the control flow example where we calculate the derivative of `d` with respect to `a`, what would happen if we changed the variable `a` to a random vector or a matrix? At this point, the result of the calculation `f(a)` is no longer a scalar. What happens to the result? How do we analyze this?
1. Let $f(x) = \sin(x)$. Plot the graph of $f$ and of its derivative $f'$. Do not exploit the fact that $f'(x) = \cos(x)$ but rather use automatic differentiation to gete the result. 
1. Let $f(x) = ((\log x^2) \cdot \sin x) + x^{-1}$. Write out a dependency graph tracing results from $x$ to $f(x)$. 
1. Use the chain rule to compute the derivative $\frac{df}{dx}$ of the aforementioned function, placing each term on the dependency graph that you constructed previously. 
1. Given the graph and the intermediate derivative results, you have a number of options when computing the gradient. Evaluate the result once starting from $x$ to $f$ and once from $f$ tracing back to $x$. The path from $x$ to $f$ is commonly known as *forward differentiation*, whereas the path from $f$ to $x$ is known as backwards differentiation. 
1. When might you want to use forward differentiation and when backwards differentiation? Hint: consider the amount of intermediate data needed, the ability to parallelize steps, and the size of matrices and vectors involved. 

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/34)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/35)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/200)
:end_tab:

```{.python .input}

```
