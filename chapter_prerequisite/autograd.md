# Automatic Gradient

In deep learning, we often need to find the gradient of a function. This section describes how to use the `autograd` package provided by MXNet to automatically find the gradient. If you are unfamiliar with the mathematical concepts (such as gradients) in this section, you can refer to the [“Mathematical Basics”](../chapter_appendix/math.md) section in the appendix.

```{.python .input  n=2}
from mxnet import autograd, nd
```

## Simple Examples

Let's look at a simple example: find the gradient of the function $y = 2\boldsymbol{x}^{\top}\boldsymbol{x}$ with respect to the column vector $\boldsymbol{x}$. Firstly, we create the variable `x` and assign an initial value.

```{.python .input}
x = nd.arange(4).reshape((4, 1))
x
```

To find the gradient of the variable `x`, we need to call the `attach_grad` function to apply for the necessary memory to store the gradient.

```{.python .input}
x.attach_grad()
```

Next, we define the function with respect to the variable `x`. To reduce computational and memory usage, MXNet does not record calculations for gradients by default. We need to call the `record` function to ask the MXNet to record the calculations related to the gradient.

```{.python .input}
with autograd.record():
    y = 2 * nd.dot(x.T, x)
```

Since the shape of `x` is (4, 1), `y` a scalar. Next, we can automatically find the gradient by calling the `backward` function. It should be noted that if `y` is not a scalar, MXNet will first sum the elements in `y` to get the new variable by default, and then find the gradient of the variable with respect to `x`.

```{.python .input}
y.backward()
```

The gradient of the function $y = 2\boldsymbol{x}^{\top}\boldsymbol{x}$ with respect to $\boldsymbol{x}$ should be $4\boldsymbol{x}$. Now let's verify that the gradient produced is correct.

```{.python .input}
assert (x.grad - 4 * x).norm().asscalar() == 0
x.grad
```

## Training Mode and Prediction Mode

As you can see from the above, after calling the `record` function, MXNet will record and calculate the gradient. In addition, `autograd` will also change the running mode from the prediction mode to the training mode by default. This can be viewed by calling the `is_training` function.

```{.python .input}
print(autograd.is_training())
with autograd.record():
    print(autograd.is_training())
```

In some cases, the same model behaves differently in the training and prediction modes. We will cover these differences in detail in later chapters.


## Find Gradient of Python Control Flow

One benefit of using MXNet is that even if the computational graph of the function contains Python's control flow (such as conditional and loop control), we may still be able to find the gradient of a variable.

Consider the following program, containing Python's conditional and loop control. It should be emphasized that the number of iterations of the loop (while loop) and the execution of the conditional judgment (if statement) depend on the value of the input `b`.

```{.python .input  n=3}
def f(a):
    b = a * 2
    while b.norm().asscalar() < 1000:
        b = b * 2
    if b.sum().asscalar() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

As previously stated, we still use the `record` function to record the calculation, and call the `backward` function to find the gradient.

```{.python .input  n=5}
a = nd.random.normal(shape=1)
a.attach_grad()
with autograd.record():
    c = f(a)
c.backward()
```

Let's analyze the `f` function defined above. Given an arbitrary input `a`, its output must be in the form of `f(a) = x * a`, where the value of the scalar coefficient `x` depends on the input `a`. Since `c = f(a)` has a gradient of `x` with respect to `a` and the value is `c / a`, we can verify the correctness of the gradient of the control flow result in the following example. 

```{.python .input  n=8}
a.grad == c / a
```

## Summary

* MXNet provides an `autograd` package to automate the derivation process.
* MXNet's `autograd` package can be used to derive general imperative programs.
* The running modes of MXNet include the training mode and the prediction mode. We can determine the running mode by `autograd.is_training()`.

## exercise

* In the example, finding the gradient of the control flow shown in this section, the variable `a` is changed to a random vector or matrix. At this point, the result of the calculation `c` is no longer a scalar. So, what will happen to the running result? How do we analyze the result?
* Redesign an example of finding the gradient of the control flow. Run and analyze the result.


## Scan the QR code to get to the [forum](https://discuss.gluon.ai/t/topic/744)

![](../img/qr_autograd.svg)
