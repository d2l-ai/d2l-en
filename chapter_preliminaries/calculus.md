# Calculus
:label:`sec_calculus`

Finding the area of a circle remained a mystery until in ancient Greece mathematicians decided to inscribe a circle with polygons of increasing number of vertices: Consider :numref:`fig_circle_area`. For a polygon with $n$ vertices we obtain $n$ triangles. The height of each triangle approaches the radius $r$ as we partition the circle more finely. At the same time, the base of it approaches $2 \pi r/n$, since the ratio between arc and secant approaches 1 for a large number of vertices. As such, the area approaches $n \cdot r \cdot \frac{1}{2} (2 \pi r/n) = \pi r^2$. 

![Finding the area of a circle as a limit procedure.](../img/polygon-circle.svg)
:label:`fig_circle_area`

This limiting procedure leads to both *differential calculus* and *integral calculus* (:numref:`sec_integral_calculus`). The former can help us solve optimization problems for finding *the best* set of parameters. Such problems are ubiquitous in deep learning. After all, in order to do well, we *train* models, updating them successively to perform better on the data that we are given for training. This is largely a problem of *optimization*. 

Note, though, that ultimately our goal is to do well on new data. The latter also involves the problem of *generalization* from previously seen to unseen data. We relegate this discussion to the following chapters when we design effective models for deep learning. In what follows, we give you a minimalist primer on differential calculus. 

## Derivatives and Differentiation

Computing derivatives is a crucial step in nearly all deep learning optimization algorithms.
To facilitate that, we typically choose loss functions
that are differentiable with respect to the model parameters.
Put simply, computing derivatives means that for each parameter,
we can accurately determine how rapidly the loss would increase or decrease,
were we to *increase* or *decrease* that parameter
by an infinitesimally small amount.

Suppose that we have a function $f: \mathbb{R} \rightarrow \mathbb{R}$,
whose input and output are both scalars.
[**The *derivative* of $f$ is defined as**]

(**$$f'(x) = \lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h},$$**)
:eqlabel:`eq_derivative`

if this limit exists.
If $f'(a)$ exists,
$f$ is said to be *differentiable* at $a$.
If $f$ is differentiable everywhere on a set, say $[a,b]$, 
then $f$ is referred to as differentiable on this set.
We can interpret the derivative $f'(x)$ in :eqref:`eq_derivative`
as the *instantaneous* rate of change of $f(x)$
with respect to $x$.
For more intuition let us experiment with an example.
(**Define $u = f(x) = 3x^2-4x$.**)

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mxnet import np, npx
npx.set_np()

def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import numpy as np

def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import numpy as np

def f(x):
    return 3 * x ** 2 - 4 * x
```

[**By setting $x=1$ and letting $h$ approach $0$,
the numerical result of $\frac{f(x+h) - f(x)}{h}$**]
in :eqref:`eq_derivative`
(**approaches $2$.**)
Though this experiment is not a mathematical proof,
we will see later see that indeed $f'(1) = 2$.

```{.python .input}
#@tab all
for h in 10.0**np.arange(-1, -6, -1):
    print(f'h={h:.5f}, numerical limit={(f(1+h)-f(1))/h:.5f}')
```

Let us familiarize ourselves with a few equivalent notations for derivatives.
Given $y = f(x)$ the following expressions are equivalent:

$$f'(x) = y' = \frac{dy}{dx} = \frac{df}{dx} = \frac{d}{dx} f(x) = Df(x) = D_x f(x),$$

where symbols $\frac{d}{dx}$ and $D$ are *differentiation operators*.
We can use the following rules to differentiate common functions:

$$\begin{aligned} \frac{d}{dx} C & = 0 && \text{$C$ is a constant} \\ \frac{d}{dx} x^n & = n x^{n-1} && \text{for } n \neq 0 \\ \frac{d}{dx} e^x & = e^x \\ \frac{d}{dx} \ln x & = x^{-1} \end{aligned}$$

To differentiate functions obtained by composition of the above (and similar functions) the following rules are handy. We assume below that $f$ and $g$ are both differentiable and that $c$ is constant. We have

$$\begin{aligned} \frac{d}{dx} [C f(x)] & = C \frac{d}{dx} f(x) && \text{Constant multiple rule} \\ \frac{d}{dx} [f(x) + g(x)] & = \frac{d}{dx} f(x) + \frac{d}{dx} g(x) && \text{Sum rule} \\ \frac{d}{dx} [f(x) g(x)] & = f(x) \frac{d}{dx} g(x) + g(x) \frac{d}{dx} f(x) && \text{Product rule} \\ \frac{d}{dx} \frac{f(x)}{g(x)} & = \frac{g(x) \frac{d}{dx} f(x) - f(x) \frac{d}{dx} g(x)}{g^2(x)} && \text{Quotient rule} \end{aligned}$$

Using this we can apply the rules to find the derivative of $3 x^2 - 4x$ via
$$\frac{d}{dx} [3 x^2 - 4x] = 3 \frac{d}{dx} x^2 - 4 \frac{d}{dx} x = 6x - 4.$$
Plugging in $x = 1$ shows that, indeed, the derivative is $2$ at this location. Note that by construction derivatives tell us the *slope* of a function at a particular location.  

## Visualization Utilities

[**To visualize slopes defined by derivatives we will use `matplotlib`,**] a popular plotting library in Python.
We need to define a few functions. As its name indicates, `use_svg_display` tells `matplotlib` to output graphics in SVG format for crisper images. The comment `#@save` is a special modifier that allows us to save the subsequent function, class, or statements in the `d2l` package such that it can be invoked later without being redefined, e.g., via `d2l.use_svg_display()`. 

```{.python .input}
#@tab all
def use_svg_display():  #@save
    """Use the svg format to display a plot in Jupyter."""
    display.set_matplotlib_formats('svg')
```

The eponymous `set_figsize` function specifies figure sizes. We use `d2l.plt` as plotting funtion since since the import statement `from matplotlib import pyplot as plt` was marked via `#@save` in the `d2l` package in the preface.

```{.python .input}
#@tab all
def set_figsize(figsize=(3.5, 2.5)):  #@save
    """Set the figure size for matplotlib."""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize
```

The following `set_axes` function sets properties of axes of figures produced by `matplotlib`.

```{.python .input}
#@tab all
#@save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib."""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim),     axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()
```

With these three functions for figure configurations,
we define the `plot` function to plot multiple curves succinctly
since we will need to visualize many curves throughout the book.
Much of the work goes into ensuring that sizes and shapes of inputs match. 

```{.python .input}
#@tab all
#@save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=[], xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot data points."""

    def has_one_axis(X):  # True if `X` (tensor or list) has 1 axis
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))
    
    if has_one_axis(X): X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
        
    set_figsize(figsize)
    if axes is None: axes = d2l.plt.gca()
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x,y,fmt) if len(x) else axes.plot(y,fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
```

Now we can [**plot the function $u = f(x)$ and its tangent line $y = 2x - 3$ at $x=1$**], where the coefficient $2$ is the slope of the tangent line.

```{.python .input}
#@tab all
x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
```

## Partial Derivatives and Gradients
:label:`subsec_calculus-grad`

Back to math. So far we dealt with the differentiation of functions of just one variable.
In deep learning, functions often depend on *many* variables.
Thus, we need to extend the ideas of differentiation to *multivariate* functions.

Let $y = f(x_1, x_2, \ldots, x_n)$ be a function with $n$ variables. The *partial derivative* of $y$ with respect to its $i^\mathrm{th}$  parameter $x_i$ is

$$ \frac{\partial y}{\partial x_i} = \lim_{h \rightarrow 0} \frac{f(x_1, \ldots, x_{i-1}, x_i+h, x_{i+1}, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}.$$


To calculate $\frac{\partial y}{\partial x_i}$, we can simply treat $x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_n$ as constants and calculate the derivative of $y$ with respect to $x_i$.
For notation of partial derivatives, the following are equivalent:

$$\frac{\partial y}{\partial x_i} = \frac{\partial f}{\partial x_i} = \partial_{x_i} f = \partial_i f = f_{x_i} = f_i = D_i f = D_{x_i} f.$$

We can concatenate partial derivatives of a multivariate function with respect to all its variables to obtain the *gradient* vector of the function.
Suppose that the input of function $f: \mathbb{R}^n \rightarrow \mathbb{R}$ is an $n$-dimensional vector $\mathbf{x} = [x_1, x_2, \ldots, x_n]^\top$ and the output is a scalar. The gradient of the function $f$ with respect to $\mathbf{x}$ is a vector of $n$ partial derivatives:

$$\nabla_{\mathbf{x}} f(\mathbf{x}) = \left[\partial_{x_1} f(\mathbf{x}), \partial_{x_2} f(\mathbf{x}), \ldots
\partial_{x_n} f(\mathbf{x})\right]^\top.$$ 

Here $\nabla_{\mathbf{x}} f(\mathbf{x})$ is typically replaced by $\nabla f(\mathbf{x})$ when there is no ambiguity.
The following rules are often used when differentiating multivariate functions:

* For all $\mathbf{A} \in \mathbb{R}^{m \times n}$ we have $\nabla_{\mathbf{x}} \mathbf{A} \mathbf{x} = \mathbf{A}^\top$ and $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A}  = \mathbf{A}$.
* For square matrices $\mathbf{A} \in \mathbb{R}^{n \times n}$ we have that $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} \mathbf{x}  = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$ and in particular
$\nabla_{\mathbf{x}} \|\mathbf{x} \|^2 = \nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{x} = 2\mathbf{x}$.

Similarly, for any matrix $\mathbf{X}$, we have $\nabla_{\mathbf{X}} \|\mathbf{X} \|_F^2 = 2\mathbf{X}$. 

## Chain Rule

Gradients can be hard to find in deep learning, since most interesting functions are concatenations of other functions. As such we may not apply any of the aforementioned rules to differentiate these functions.
Fortunately, the *chain rule* (in Latin 'catena' means chain) takes care of this. Let's start with a single variable. 
Suppose that functions $y=f(u)$ and $u=g(x)$ are both differentiable, then the chain rule states that for $y = f(g(x))$ we have

$$\frac{dy}{dx} = \frac{dy}{du} \frac{du}{dx}.$$

Next consider functions with an arbitrary number of variables.
Suppose that $y = f(\mathbf{u})$ has variables
$u_1, u_2, \ldots, u_m$, where each $u_i = g_i(\mathbf{x})$ 
has variables $x_1, x_2, \ldots, x_n$, in short 
$\mathbf{u} = g(\mathbf{x})$. Then the chain rule reads as follows:

$$\frac{dy}{dx_i} = \frac{dy}{du_1} \frac{du_1}{dx_i} + \frac{dy}{du_2} \frac{du_2}{dx_i} + \cdots + \frac{dy}{du_m} \frac{du_m}{dx_i} \text{ and thus }
\nabla_{\mathbf{x}} y = \nabla_{\mathbf{u}} y \cdot \nabla_{x} \mathbf{u}$$

Note that $\nabla_{x} \mathbf{u}$ is a *matrix* since it contains the derivative of a vector with regard to a vector. As such, evaluating the gradient requires us to compute a vector-matrix product. This is one of the key reasons why linear algebra is such an integral building block in building deep learning systems. 

## Summary

So far be barely scratched the surface of what differential calculus can do. That said, there are a number of pieces that come into focus: firstly, the composition rules for differentiation allow one to compute the derivatives of functions mechanically by rote application. Hence, there's no need for creativity. In particular, the derivatives can be computed *automatically* by an autograd library. Second, the process of computing derivatives requires us to multiply matrices as we trace the dependency graph of variables from input to output. In particular, this graph is traversed in a *forward* direction when we evaluate a function and in a *backwards* direction when we compute gradients. Later chapters will formalize this as backpropagation. 

From the viewpoint of optimization, gradients allow us to identify how we should be changing parameters of a model in such a way that it performs better, e.g., by reducing the value of a loss function or by increasing the reward associated with an action. This forms the centerpiece of many optimization algorithms that we will encounter throughout this book. 

## Exercises

1. So far we took the rules for derivatives for granted. Using the definition and limits prove the properties for a) $f(x) = c$, b) $f(x) = x^n$, c) $f(x) = e^x$ and d) $f(x) = \ln x$.
1. In the same vein, prove the product, sum, and quotient rule from first principles. 
1. Prove that the constant multiple rule follows as a special case of the product rule. 
1. Calculate the derivative of $f(x) = x^x$. 
1. What does it mean that $f'(x) = 0$ for some $x$? Give an example of a function $f$ and a location $x$ for which this might hold. 
1. Plot the function $y = f(x) = x^3 - \frac{1}{x}$ and plot its tangent line at $x = 1$.
1. Find the gradient of the function $f(\mathbf{x}) = 3x_1^2 + 5e^{x_2}$.
1. What is the gradient of the function $f(\mathbf{x}) = \|\mathbf{x}\|_2$? What happens for $\mathbf{x} = 0$?
1. Can you write out the chain rule for the case where $u = f(x, y, z)$ and $x = x(a, b)$, $y = y(a, b)$, and $z = z(a, b)$?
1. Given a function $f(x)$ that is invertible, compute the derivative of its inverse $f^{-1}(x)$. Here we have that $f^{-1}(f(x)) = x$ and conversely $f(f^{-1}(y)) = y$. Hint: use these properties in your derivation. 

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/32)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/33)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/197)
:end_tab:

```{.python .input}

```
