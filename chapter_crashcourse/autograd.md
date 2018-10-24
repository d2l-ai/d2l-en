# Automatic Differentiation

In machine learning, we *train* models to get better and better as a function of experience. Usually, *getting better* means minimizing a *loss function*, i.e. a score that answers "how *bad* is our model?" With neural networks, we choose loss functions to be differentiable with respect to our parameters. Put simply, this means that for each of the model's parameters, we can determine how much *increasing* or *decreasing* it might affect the loss. While the calculations are straightforward, for complex models, working it out by hand can be a pain (and often error-prone).

The autograd package expedites this work by automatically calculating derivatives. And while most other libraries require that we compile a symbolic graph to take automatic derivatives, `autograd` allows you to take derivatives while writing  ordinary imperative code. Every time you make pass through your model, `autograd` builds a graph on the fly, through which it can immediately backpropagate gradients. If you are unfamiliar with some of the math, e.g. gradients, please refer to the [“Mathematical Basics”](../chapter_appendix/math.md) section in the appendix.

```{.python .input  n=1}
from mxnet import autograd, nd
```

## A Simple Example

As a toy example, let's say that we are interested in differentiating the mapping $y = 2\boldsymbol{x}^{\top}\boldsymbol{x}$ with respect to the column vector $\boldsymbol{x}$. Firstly, we create the variable `x` and assign an initial value.

```{.python .input  n=2}
x = nd.arange(4).reshape((4, 1))
print(x)
```

Once we compute the gradient of ``y`` with respect to ``x``, we'll need a place to store it. We can tell an NDArray that we plan to store a gradient by invoking its ``attach_grad()`` method.

```{.python .input  n=3}
x.attach_grad()
```

Now we're going to compute ``y`` and MXNet will generate a computation graph on the fly. It's as if MXNet turned on a recording device and captured the exact path by which each variable was generated.

Note that building the computation graph requires a nontrivial amount of computation. So MXNet will *only* build the graph when explicitly told to do so. This happens by placing code inside a ``with autograd.record():`` block.

```{.python .input  n=4}
with autograd.record():
    y = 2 * nd.dot(x.T, x)
print(y)
```

Since the shape of `x` is (4, 1), `y` is a scalar. Next, we can automatically find the gradient by calling the `backward` function. It should be noted that if `y` is not a scalar, MXNet will first sum the elements in `y` to get the new variable by default, and then find the gradient of the variable with respect to `x`.

```{.python .input  n=5}
y.backward()
```

The gradient of the function $y = 2\boldsymbol{x}^{\top}\boldsymbol{x}$ with respect to $\boldsymbol{x}$ should be $4\boldsymbol{x}$. Now let's verify that the gradient produced is correct.

```{.python .input  n=6}
print((x.grad - 4 * x).norm().asscalar() == 0)
print(x.grad)
```

## Training Mode and Prediction Mode

As you can see from the above, after calling the `record` function, MXNet will record and calculate the gradient. In addition, `autograd` will also change the running mode from the prediction mode to the training mode by default. This can be viewed by calling the `is_training` function.

```{.python .input  n=7}
print(autograd.is_training())
with autograd.record():
    print(autograd.is_training())
```

In some cases, the same model behaves differently in the training and prediction modes (such as batch normalization). In other cases, some models may store more auxiliary variables to make computing gradients easier. We will cover these differences in detail in later chapters. For now, you need not worry about these details just yet.

## Computing the Gradient of Python Control Flow

One benefit of using automatic differentiation is that even if the computational graph of the function contains Python's control flow (such as conditional and loop control), we may still be able to find the gradient of a variable. Consider the following program:  It should be emphasized that the number of iterations of the loop (while loop) and the execution of the conditional judgment (if statement) depend on the value of the input `b`.

```{.python .input  n=8}
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

Note that the number of iterations of the while loop and the execution of the conditional statement (if then else) depend on the value of `a`. To compute gradients, we need to `record` the calculation, and call the `backward` function to find the gradient.

```{.python .input  n=9}
a = nd.random.normal(shape=1)
a.attach_grad()
with autograd.record():
    d = f(a)
d.backward()
```

Let's analyze the `f` function defined above. As you can see, it is piecewise linear in its input `a`. In other words, for any `a` there exists some constant such that for a given range `f(a) = g * a`. Consequently `d / a` allows us to verify that the gradient is correct:

```{.python .input  n=10}
print(a.grad == (d / a))
```

## Head gradients and the chain rule

*Caution: This part is tricky and not necessary to understanding subsequent sections. That said, it is needed if you want to build new layers from scratch. You can skip this on a first read.*

Sometimes when we call the backward method, e.g. `y.backward()`, where `y` is a function of `x` we are just interested in the derivative of `y` with respect to `x`. Mathematicians write this as $\frac{dy(x)}{dx}$. At other times, we may be interested in the gradient of `z` with respect to `x`, where `z` is a function of `y`, which in turn, is a function of `x`. That is, we are interested in $\frac{d}{dx} z(y(x))$. Recall that by the chain rule
$$\frac{d}{dx} z(y(x)) = \frac{dz(y)}{dy} \frac{dy(x)}{dx}.$$
So, when ``y`` is part of a larger function ``z``, and we want ``x.grad`` to store $\frac{dz}{dx}$, we can pass in the *head gradient* $\frac{dz}{dy}$ as an input to ``backward()``. The default argument is ``nd.ones_like(y)``. See [Wikipedia](https://en.wikipedia.org/wiki/Chain_rule) for more details.

```{.python .input  n=11}
with autograd.record():
    y = x * 2
    z = y * x

head_gradient = nd.array([10, 1., .1, .01])
z.backward(head_gradient)
print(x.grad)
```

## Summary

* MXNet provides an `autograd` package to automate the derivation process.
* MXNet's `autograd` package can be used to derive general imperative programs.
* The running modes of MXNet include the training mode and the prediction mode. We can determine the running mode by `autograd.is_training()`.

## Exercises

* In the example, finding the gradient of the control flow shown in this section, the variable `a` is changed to a random vector or matrix. At this point, the result of the calculation `c` is no longer a scalar. What happens to the result. How do we analyze this?
* Redesign an example of finding the gradient of the control flow. Run and analyze the result.
* In a second price auction (such as in eBay or in computational advertising) the winning bidder pays the second highest price. Compute the gradient of the winning bidder with regard to his bid using `autograd`. Why do you get a pathological result? What does this tell us about the mechanism? For more details read the paper by [Edelman, Ostrovski and Schwartz, 2005](https://www.benedelman.org/publications/gsp-060801.pdf).
* Why is the second derivative much more expensive to compute than the first derivative?
* Derive the head gradient relationship for the chain rule. If you get stuck, use the  [Wikipedia Chain Rule](https://en.wikipedia.org/wiki/Chain_rule) entry.
* Assume $f(x) = \sin(x)$. Plot $f(x)$ and $\frac{df(x)}{dx}$ on a graph, where you computed the latter without any symbolic calculations, i.e. without exploiting that $f'(x) = \cos(x)$.

## Scan the QR code to get to the [forum](https://discuss.gluon.ai/t/topic/744)

![](../img/qr_autograd.svg)
