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

```{.python .input  n=1}
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

```{.python .input  n=15}
#@tab all
for h in 10.0**np.arange(-1, -6, -1):
    print(f'h={h:.5f}, numerical limit={(f(1+h)-f(1))/h:.5f}')
```

```{.json .output n=15}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "h=0.10000, numerical limit=2.30000\nh=0.01000, numerical limit=2.03000\nh=0.00100, numerical limit=2.00300\nh=0.00010, numerical limit=2.00030\nh=0.00001, numerical limit=2.00003\n"
 }
]
```

Let us familiarize ourselves with a few equivalent notations for derivatives.
Given $y = f(x)$ the following expressions are equivalent:

$$f'(x) = y' = \frac{dy}{dx} = \frac{df}{dx} = \frac{d}{dx} f(x) = Df(x) = D_x f(x),$$

where symbols $\frac{d}{dx}$ and $D$ are *differentiation operators*.
We can use the following rules to differentiate common functions:
$$\begin{align}
\frac{d}{dx} C & = 0 && \text{$C$ is a constant} \\
\frac{d}{dx} x^n & = n x^{n-1} && \text{for } n \neq 0 \\
\frac{d}{dx} e^x & = e^x \\
\frac{d}{dx} \ln x & = x^{-1}
\end{align}$$
To differentiate functions obtained by composition of the above (and similar functions) the following rules are handy. We assume below that $f$ and $g$ are both differentiable and that $c$ is constant. We have
$$\begin{align}
\frac{d}{dx} [C f(x)] & = C \frac{d}{dx} f(x) && \text{Constant multiple rule} \\
\frac{d}{dx} [f(x) + g(x)] & = \frac{d}{dx} f(x) + \frac{d}{dx} g(x) && \text{Sum rule} \\
\frac{d}{dx} [f(x) g(x)] & = f(x) \frac{d}{dx} g(x) + g(x) \frac{d}{dx} f(x) && \text{Product rule} \\
\frac{d}{dx} \frac{f(x)}{g(x)} & = \frac{g(x) \frac{d}{dx} f(x) - f(x) \frac{d}{dx} g(x)}{g^2(x)} && \text{Quotient rule}
\end{align}$$
Using this we can apply the rules to find the derivative of $3 x^2 - 4x$ via
$$\frac{d}{dx} [3 x^2 - 4x] = 3 \frac{d}{dx} x^2 - 4 \frac{d}{dx} x = 6x - 4.$$
Plugging in $x = 1$ shows that, indeed, the derivative is $2$ at this location. Note that by construction derivatives tell us the *slope* of a function at a particular location.  

## Visualization Utilities

[**To visualize slopes defined by derivatives we will use `matplotlib`,**] a popular plotting library in Python.
We need to define a few functions. As its name indicates, `use_svg_display` tells `matplotlib` to output graphics in SVG format for crisper images. The comment `#@save` is a special modifier that allows us to save the subsequent function, class, or statements in the `d2l` package such that it can be invoked later without being redefined, e.g., via `d2l.use_svg_display()`. 

```{.python .input  n=30}
#@tab all
def use_svg_display():  #@save
    """Use the svg format to display a plot in Jupyter."""
    display.set_matplotlib_formats('svg')
```

The eponymous `set_figsize` function specifies figure sizes. We use `d2l.plt` as plotting funtion since since the import statement `from matplotlib import pyplot as plt` was marked via `#@save` in the `d2l` package in the preface.

```{.python .input  n=17}
#@tab all
def set_figsize(figsize=(3.5, 2.5)):  #@save
    """Set the figure size for matplotlib."""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize
```

The following `set_axes` function sets properties of axes of figures produced by `matplotlib`.

```{.python .input  n=18}
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

```{.python .input  n=57}
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

```{.python .input  n=58}
#@tab all
x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
```

```{.json .output n=58}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "/var/folders/7z/v7dyvssd1q99sbzh8ttw5yxr0000gs/T/ipykernel_82365/540752353.py:4: DeprecationWarning: `set_matplotlib_formats` is deprecated since IPython 7.23, directly use `matplotlib_inline.backend_inline.set_matplotlib_formats()`\n  display.set_matplotlib_formats('svg')\n"
 },
 {
  "data": {
   "image/svg+xml": "<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"no\"?>\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n  \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n<svg height=\"184.15625pt\" version=\"1.1\" viewBox=\"0 0 246.603125 184.15625\" width=\"246.603125pt\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n <metadata>\n  <rdf:RDF xmlns:cc=\"http://creativecommons.org/ns#\" xmlns:dc=\"http://purl.org/dc/elements/1.1/\" xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n   <cc:Work>\n    <dc:type rdf:resource=\"http://purl.org/dc/dcmitype/StillImage\"/>\n    <dc:date>2021-07-27T17:50:43.608776</dc:date>\n    <dc:format>image/svg+xml</dc:format>\n    <dc:creator>\n     <cc:Agent>\n      <dc:title>Matplotlib v3.4.2, https://matplotlib.org/</dc:title>\n     </cc:Agent>\n    </dc:creator>\n   </cc:Work>\n  </rdf:RDF>\n </metadata>\n <defs>\n  <style type=\"text/css\">*{stroke-linecap:butt;stroke-linejoin:round;}</style>\n </defs>\n <g id=\"figure_1\">\n  <g id=\"patch_1\">\n   <path d=\"M 0 184.15625 \nL 246.603125 184.15625 \nL 246.603125 -0 \nL 0 -0 \nz\n\" style=\"fill:none;\"/>\n  </g>\n  <g id=\"axes_1\">\n   <g id=\"patch_2\">\n    <path d=\"M 40.603125 146.6 \nL 235.903125 146.6 \nL 235.903125 10.7 \nL 40.603125 10.7 \nz\n\" style=\"fill:#ffffff;\"/>\n   </g>\n   <g id=\"matplotlib.axis_1\">\n    <g id=\"xtick_1\">\n     <g id=\"line2d_1\">\n      <path clip-path=\"url(#p3ebdaefcff)\" d=\"M 49.480398 146.6 \nL 49.480398 10.7 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_2\">\n      <defs>\n       <path d=\"M 0 0 \nL 0 3.5 \n\" id=\"m7b7e66fa46\" style=\"stroke:#000000;stroke-width:0.8;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"49.480398\" xlink:href=\"#m7b7e66fa46\" y=\"146.6\"/>\n      </g>\n     </g>\n     <g id=\"line2d_3\">\n      <defs>\n       <path d=\"M 0 0 \nL 0 -3.5 \n\" id=\"m7bfd479842\" style=\"stroke:#000000;stroke-width:0.8;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"49.480398\" xlink:href=\"#m7bfd479842\" y=\"10.7\"/>\n      </g>\n     </g>\n     <g id=\"text_1\">\n      <!-- 0 -->\n      <g transform=\"translate(46.299148 161.198437)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 2034 4250 \nQ 1547 4250 1301 3770 \nQ 1056 3291 1056 2328 \nQ 1056 1369 1301 889 \nQ 1547 409 2034 409 \nQ 2525 409 2770 889 \nQ 3016 1369 3016 2328 \nQ 3016 3291 2770 3770 \nQ 2525 4250 2034 4250 \nz\nM 2034 4750 \nQ 2819 4750 3233 4129 \nQ 3647 3509 3647 2328 \nQ 3647 1150 3233 529 \nQ 2819 -91 2034 -91 \nQ 1250 -91 836 529 \nQ 422 1150 422 2328 \nQ 422 3509 836 4129 \nQ 1250 4750 2034 4750 \nz\n\" id=\"DejaVuSans-30\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-30\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_2\">\n     <g id=\"line2d_4\">\n      <path clip-path=\"url(#p3ebdaefcff)\" d=\"M 110.702968 146.6 \nL 110.702968 10.7 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_5\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"110.702968\" xlink:href=\"#m7b7e66fa46\" y=\"146.6\"/>\n      </g>\n     </g>\n     <g id=\"line2d_6\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"110.702968\" xlink:href=\"#m7bfd479842\" y=\"10.7\"/>\n      </g>\n     </g>\n     <g id=\"text_2\">\n      <!-- 1 -->\n      <g transform=\"translate(107.521718 161.198437)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 794 531 \nL 1825 531 \nL 1825 4091 \nL 703 3866 \nL 703 4441 \nL 1819 4666 \nL 2450 4666 \nL 2450 531 \nL 3481 531 \nL 3481 0 \nL 794 0 \nL 794 531 \nz\n\" id=\"DejaVuSans-31\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-31\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_3\">\n     <g id=\"line2d_7\">\n      <path clip-path=\"url(#p3ebdaefcff)\" d=\"M 171.925539 146.6 \nL 171.925539 10.7 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_8\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"171.925539\" xlink:href=\"#m7b7e66fa46\" y=\"146.6\"/>\n      </g>\n     </g>\n     <g id=\"line2d_9\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"171.925539\" xlink:href=\"#m7bfd479842\" y=\"10.7\"/>\n      </g>\n     </g>\n     <g id=\"text_3\">\n      <!-- 2 -->\n      <g transform=\"translate(168.744289 161.198437)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 1228 531 \nL 3431 531 \nL 3431 0 \nL 469 0 \nL 469 531 \nQ 828 903 1448 1529 \nQ 2069 2156 2228 2338 \nQ 2531 2678 2651 2914 \nQ 2772 3150 2772 3378 \nQ 2772 3750 2511 3984 \nQ 2250 4219 1831 4219 \nQ 1534 4219 1204 4116 \nQ 875 4013 500 3803 \nL 500 4441 \nQ 881 4594 1212 4672 \nQ 1544 4750 1819 4750 \nQ 2544 4750 2975 4387 \nQ 3406 4025 3406 3419 \nQ 3406 3131 3298 2873 \nQ 3191 2616 2906 2266 \nQ 2828 2175 2409 1742 \nQ 1991 1309 1228 531 \nz\n\" id=\"DejaVuSans-32\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-32\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_4\">\n     <g id=\"line2d_10\">\n      <path clip-path=\"url(#p3ebdaefcff)\" d=\"M 233.148109 146.6 \nL 233.148109 10.7 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_11\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"233.148109\" xlink:href=\"#m7b7e66fa46\" y=\"146.6\"/>\n      </g>\n     </g>\n     <g id=\"line2d_12\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"233.148109\" xlink:href=\"#m7bfd479842\" y=\"10.7\"/>\n      </g>\n     </g>\n     <g id=\"text_4\">\n      <!-- 3 -->\n      <g transform=\"translate(229.966859 161.198437)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 2597 2516 \nQ 3050 2419 3304 2112 \nQ 3559 1806 3559 1356 \nQ 3559 666 3084 287 \nQ 2609 -91 1734 -91 \nQ 1441 -91 1130 -33 \nQ 819 25 488 141 \nL 488 750 \nQ 750 597 1062 519 \nQ 1375 441 1716 441 \nQ 2309 441 2620 675 \nQ 2931 909 2931 1356 \nQ 2931 1769 2642 2001 \nQ 2353 2234 1838 2234 \nL 1294 2234 \nL 1294 2753 \nL 1863 2753 \nQ 2328 2753 2575 2939 \nQ 2822 3125 2822 3475 \nQ 2822 3834 2567 4026 \nQ 2313 4219 1838 4219 \nQ 1578 4219 1281 4162 \nQ 984 4106 628 3988 \nL 628 4550 \nQ 988 4650 1302 4700 \nQ 1616 4750 1894 4750 \nQ 2613 4750 3031 4423 \nQ 3450 4097 3450 3541 \nQ 3450 3153 3228 2886 \nQ 3006 2619 2597 2516 \nz\n\" id=\"DejaVuSans-33\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-33\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"text_5\">\n     <!-- x -->\n     <g transform=\"translate(135.29375 174.876562)scale(0.1 -0.1)\">\n      <defs>\n       <path d=\"M 3513 3500 \nL 2247 1797 \nL 3578 0 \nL 2900 0 \nL 1881 1375 \nL 863 0 \nL 184 0 \nL 1544 1831 \nL 300 3500 \nL 978 3500 \nL 1906 2253 \nL 2834 3500 \nL 3513 3500 \nz\n\" id=\"DejaVuSans-78\" transform=\"scale(0.015625)\"/>\n      </defs>\n      <use xlink:href=\"#DejaVuSans-78\"/>\n     </g>\n    </g>\n   </g>\n   <g id=\"matplotlib.axis_2\">\n    <g id=\"ytick_1\">\n     <g id=\"line2d_13\">\n      <path clip-path=\"url(#p3ebdaefcff)\" d=\"M 40.603125 118.135514 \nL 235.903125 118.135514 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_14\">\n      <defs>\n       <path d=\"M 0 0 \nL -3.5 0 \n\" id=\"m4f7de2c8ba\" style=\"stroke:#000000;stroke-width:0.8;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"40.603125\" xlink:href=\"#m4f7de2c8ba\" y=\"118.135514\"/>\n      </g>\n     </g>\n     <g id=\"line2d_15\">\n      <defs>\n       <path d=\"M 0 0 \nL 3.5 0 \n\" id=\"m0f6f7541ce\" style=\"stroke:#000000;stroke-width:0.8;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"235.903125\" xlink:href=\"#m0f6f7541ce\" y=\"118.135514\"/>\n      </g>\n     </g>\n     <g id=\"text_6\">\n      <!-- 0 -->\n      <g transform=\"translate(27.240625 121.934732)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-30\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_2\">\n     <g id=\"line2d_16\">\n      <path clip-path=\"url(#p3ebdaefcff)\" d=\"M 40.603125 80.990157 \nL 235.903125 80.990157 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_17\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"40.603125\" xlink:href=\"#m4f7de2c8ba\" y=\"80.990157\"/>\n      </g>\n     </g>\n     <g id=\"line2d_18\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"235.903125\" xlink:href=\"#m0f6f7541ce\" y=\"80.990157\"/>\n      </g>\n     </g>\n     <g id=\"text_7\">\n      <!-- 5 -->\n      <g transform=\"translate(27.240625 84.789376)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 691 4666 \nL 3169 4666 \nL 3169 4134 \nL 1269 4134 \nL 1269 2991 \nQ 1406 3038 1543 3061 \nQ 1681 3084 1819 3084 \nQ 2600 3084 3056 2656 \nQ 3513 2228 3513 1497 \nQ 3513 744 3044 326 \nQ 2575 -91 1722 -91 \nQ 1428 -91 1123 -41 \nQ 819 9 494 109 \nL 494 744 \nQ 775 591 1075 516 \nQ 1375 441 1709 441 \nQ 2250 441 2565 725 \nQ 2881 1009 2881 1497 \nQ 2881 1984 2565 2268 \nQ 2250 2553 1709 2553 \nQ 1456 2553 1204 2497 \nQ 953 2441 691 2322 \nL 691 4666 \nz\n\" id=\"DejaVuSans-35\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-35\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_3\">\n     <g id=\"line2d_19\">\n      <path clip-path=\"url(#p3ebdaefcff)\" d=\"M 40.603125 43.844801 \nL 235.903125 43.844801 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_20\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"40.603125\" xlink:href=\"#m4f7de2c8ba\" y=\"43.844801\"/>\n      </g>\n     </g>\n     <g id=\"line2d_21\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"235.903125\" xlink:href=\"#m0f6f7541ce\" y=\"43.844801\"/>\n      </g>\n     </g>\n     <g id=\"text_8\">\n      <!-- 10 -->\n      <g transform=\"translate(20.878125 47.64402)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-31\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-30\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"text_9\">\n     <!-- f(x) -->\n     <g transform=\"translate(14.798437 87.271094)rotate(-90)scale(0.1 -0.1)\">\n      <defs>\n       <path d=\"M 2375 4863 \nL 2375 4384 \nL 1825 4384 \nQ 1516 4384 1395 4259 \nQ 1275 4134 1275 3809 \nL 1275 3500 \nL 2222 3500 \nL 2222 3053 \nL 1275 3053 \nL 1275 0 \nL 697 0 \nL 697 3053 \nL 147 3053 \nL 147 3500 \nL 697 3500 \nL 697 3744 \nQ 697 4328 969 4595 \nQ 1241 4863 1831 4863 \nL 2375 4863 \nz\n\" id=\"DejaVuSans-66\" transform=\"scale(0.015625)\"/>\n       <path d=\"M 1984 4856 \nQ 1566 4138 1362 3434 \nQ 1159 2731 1159 2009 \nQ 1159 1288 1364 580 \nQ 1569 -128 1984 -844 \nL 1484 -844 \nQ 1016 -109 783 600 \nQ 550 1309 550 2009 \nQ 550 2706 781 3412 \nQ 1013 4119 1484 4856 \nL 1984 4856 \nz\n\" id=\"DejaVuSans-28\" transform=\"scale(0.015625)\"/>\n       <path d=\"M 513 4856 \nL 1013 4856 \nQ 1481 4119 1714 3412 \nQ 1947 2706 1947 2009 \nQ 1947 1309 1714 600 \nQ 1481 -109 1013 -844 \nL 513 -844 \nQ 928 -128 1133 580 \nQ 1338 1288 1338 2009 \nQ 1338 2731 1133 3434 \nQ 928 4138 513 4856 \nz\n\" id=\"DejaVuSans-29\" transform=\"scale(0.015625)\"/>\n      </defs>\n      <use xlink:href=\"#DejaVuSans-66\"/>\n      <use x=\"35.205078\" xlink:href=\"#DejaVuSans-28\"/>\n      <use x=\"74.21875\" xlink:href=\"#DejaVuSans-78\"/>\n      <use x=\"133.398438\" xlink:href=\"#DejaVuSans-29\"/>\n     </g>\n    </g>\n   </g>\n   <g id=\"line2d_22\">\n    <path clip-path=\"url(#p3ebdaefcff)\" d=\"M 49.480398 118.135514 \nL 55.602655 120.88427 \nL 61.724912 123.187282 \nL 67.847169 125.04455 \nL 73.969426 126.456073 \nL 80.091683 127.421853 \nL 86.21394 127.941888 \nL 92.336197 128.016178 \nL 98.458454 127.644725 \nL 104.580711 126.827527 \nL 110.702968 125.564585 \nL 116.825225 123.855898 \nL 122.947482 121.701468 \nL 129.069739 119.101293 \nL 135.191996 116.055374 \nL 141.314254 112.56371 \nL 147.436511 108.626302 \nL 153.558768 104.24315 \nL 159.681025 99.414254 \nL 165.803282 94.139614 \nL 171.925539 88.419229 \nL 178.047796 82.2531 \nL 184.170053 75.641226 \nL 190.29231 68.583608 \nL 196.414567 61.080247 \nL 202.536824 53.13114 \nL 208.659081 44.73629 \nL 214.781338 35.895695 \nL 220.903595 26.609356 \nL 227.025852 16.877273 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_23\">\n    <path clip-path=\"url(#p3ebdaefcff)\" d=\"M 49.480398 140.422727 \nL 55.602655 138.936913 \nL 61.724912 137.451099 \nL 67.847169 135.965285 \nL 73.969426 134.47947 \nL 80.091683 132.993656 \nL 86.21394 131.507842 \nL 92.336197 130.022028 \nL 98.458454 128.536213 \nL 104.580711 127.050399 \nL 110.702968 125.564585 \nL 116.825225 124.078771 \nL 122.947482 122.592956 \nL 129.069739 121.107142 \nL 135.191996 119.621328 \nL 141.314254 118.135514 \nL 147.436511 116.649699 \nL 153.558768 115.163885 \nL 159.681025 113.678071 \nL 165.803282 112.192257 \nL 171.925539 110.706442 \nL 178.047796 109.220628 \nL 184.170053 107.734814 \nL 190.29231 106.249 \nL 196.414567 104.763185 \nL 202.536824 103.277371 \nL 208.659081 101.791557 \nL 214.781338 100.305743 \nL 220.903595 98.819928 \nL 227.025852 97.334114 \n\" style=\"fill:none;stroke:#bf00bf;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"patch_3\">\n    <path d=\"M 40.603125 146.6 \nL 40.603125 10.7 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_4\">\n    <path d=\"M 235.903125 146.6 \nL 235.903125 10.7 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_5\">\n    <path d=\"M 40.603125 146.6 \nL 235.903125 146.6 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_6\">\n    <path d=\"M 40.603125 10.7 \nL 235.903125 10.7 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"legend_1\">\n    <g id=\"patch_7\">\n     <path d=\"M 47.603125 48.05625 \nL 172.153125 48.05625 \nQ 174.153125 48.05625 174.153125 46.05625 \nL 174.153125 17.7 \nQ 174.153125 15.7 172.153125 15.7 \nL 47.603125 15.7 \nQ 45.603125 15.7 45.603125 17.7 \nL 45.603125 46.05625 \nQ 45.603125 48.05625 47.603125 48.05625 \nz\n\" style=\"fill:#ffffff;opacity:0.8;stroke:#cccccc;stroke-linejoin:miter;\"/>\n    </g>\n    <g id=\"line2d_24\">\n     <path d=\"M 49.603125 23.798437 \nL 69.603125 23.798437 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n    </g>\n    <g id=\"line2d_25\"/>\n    <g id=\"text_10\">\n     <!-- f(x) -->\n     <g transform=\"translate(77.603125 27.298437)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-66\"/>\n      <use x=\"35.205078\" xlink:href=\"#DejaVuSans-28\"/>\n      <use x=\"74.21875\" xlink:href=\"#DejaVuSans-78\"/>\n      <use x=\"133.398438\" xlink:href=\"#DejaVuSans-29\"/>\n     </g>\n    </g>\n    <g id=\"line2d_26\">\n     <path d=\"M 49.603125 38.476562 \nL 69.603125 38.476562 \n\" style=\"fill:none;stroke:#bf00bf;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n    </g>\n    <g id=\"line2d_27\"/>\n    <g id=\"text_11\">\n     <!-- Tangent line (x=1) -->\n     <g transform=\"translate(77.603125 41.976562)scale(0.1 -0.1)\">\n      <defs>\n       <path d=\"M -19 4666 \nL 3928 4666 \nL 3928 4134 \nL 2272 4134 \nL 2272 0 \nL 1638 0 \nL 1638 4134 \nL -19 4134 \nL -19 4666 \nz\n\" id=\"DejaVuSans-54\" transform=\"scale(0.015625)\"/>\n       <path d=\"M 2194 1759 \nQ 1497 1759 1228 1600 \nQ 959 1441 959 1056 \nQ 959 750 1161 570 \nQ 1363 391 1709 391 \nQ 2188 391 2477 730 \nQ 2766 1069 2766 1631 \nL 2766 1759 \nL 2194 1759 \nz\nM 3341 1997 \nL 3341 0 \nL 2766 0 \nL 2766 531 \nQ 2569 213 2275 61 \nQ 1981 -91 1556 -91 \nQ 1019 -91 701 211 \nQ 384 513 384 1019 \nQ 384 1609 779 1909 \nQ 1175 2209 1959 2209 \nL 2766 2209 \nL 2766 2266 \nQ 2766 2663 2505 2880 \nQ 2244 3097 1772 3097 \nQ 1472 3097 1187 3025 \nQ 903 2953 641 2809 \nL 641 3341 \nQ 956 3463 1253 3523 \nQ 1550 3584 1831 3584 \nQ 2591 3584 2966 3190 \nQ 3341 2797 3341 1997 \nz\n\" id=\"DejaVuSans-61\" transform=\"scale(0.015625)\"/>\n       <path d=\"M 3513 2113 \nL 3513 0 \nL 2938 0 \nL 2938 2094 \nQ 2938 2591 2744 2837 \nQ 2550 3084 2163 3084 \nQ 1697 3084 1428 2787 \nQ 1159 2491 1159 1978 \nL 1159 0 \nL 581 0 \nL 581 3500 \nL 1159 3500 \nL 1159 2956 \nQ 1366 3272 1645 3428 \nQ 1925 3584 2291 3584 \nQ 2894 3584 3203 3211 \nQ 3513 2838 3513 2113 \nz\n\" id=\"DejaVuSans-6e\" transform=\"scale(0.015625)\"/>\n       <path d=\"M 2906 1791 \nQ 2906 2416 2648 2759 \nQ 2391 3103 1925 3103 \nQ 1463 3103 1205 2759 \nQ 947 2416 947 1791 \nQ 947 1169 1205 825 \nQ 1463 481 1925 481 \nQ 2391 481 2648 825 \nQ 2906 1169 2906 1791 \nz\nM 3481 434 \nQ 3481 -459 3084 -895 \nQ 2688 -1331 1869 -1331 \nQ 1566 -1331 1297 -1286 \nQ 1028 -1241 775 -1147 \nL 775 -588 \nQ 1028 -725 1275 -790 \nQ 1522 -856 1778 -856 \nQ 2344 -856 2625 -561 \nQ 2906 -266 2906 331 \nL 2906 616 \nQ 2728 306 2450 153 \nQ 2172 0 1784 0 \nQ 1141 0 747 490 \nQ 353 981 353 1791 \nQ 353 2603 747 3093 \nQ 1141 3584 1784 3584 \nQ 2172 3584 2450 3431 \nQ 2728 3278 2906 2969 \nL 2906 3500 \nL 3481 3500 \nL 3481 434 \nz\n\" id=\"DejaVuSans-67\" transform=\"scale(0.015625)\"/>\n       <path d=\"M 3597 1894 \nL 3597 1613 \nL 953 1613 \nQ 991 1019 1311 708 \nQ 1631 397 2203 397 \nQ 2534 397 2845 478 \nQ 3156 559 3463 722 \nL 3463 178 \nQ 3153 47 2828 -22 \nQ 2503 -91 2169 -91 \nQ 1331 -91 842 396 \nQ 353 884 353 1716 \nQ 353 2575 817 3079 \nQ 1281 3584 2069 3584 \nQ 2775 3584 3186 3129 \nQ 3597 2675 3597 1894 \nz\nM 3022 2063 \nQ 3016 2534 2758 2815 \nQ 2500 3097 2075 3097 \nQ 1594 3097 1305 2825 \nQ 1016 2553 972 2059 \nL 3022 2063 \nz\n\" id=\"DejaVuSans-65\" transform=\"scale(0.015625)\"/>\n       <path d=\"M 1172 4494 \nL 1172 3500 \nL 2356 3500 \nL 2356 3053 \nL 1172 3053 \nL 1172 1153 \nQ 1172 725 1289 603 \nQ 1406 481 1766 481 \nL 2356 481 \nL 2356 0 \nL 1766 0 \nQ 1100 0 847 248 \nQ 594 497 594 1153 \nL 594 3053 \nL 172 3053 \nL 172 3500 \nL 594 3500 \nL 594 4494 \nL 1172 4494 \nz\n\" id=\"DejaVuSans-74\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-20\" transform=\"scale(0.015625)\"/>\n       <path d=\"M 603 4863 \nL 1178 4863 \nL 1178 0 \nL 603 0 \nL 603 4863 \nz\n\" id=\"DejaVuSans-6c\" transform=\"scale(0.015625)\"/>\n       <path d=\"M 603 3500 \nL 1178 3500 \nL 1178 0 \nL 603 0 \nL 603 3500 \nz\nM 603 4863 \nL 1178 4863 \nL 1178 4134 \nL 603 4134 \nL 603 4863 \nz\n\" id=\"DejaVuSans-69\" transform=\"scale(0.015625)\"/>\n       <path d=\"M 678 2906 \nL 4684 2906 \nL 4684 2381 \nL 678 2381 \nL 678 2906 \nz\nM 678 1631 \nL 4684 1631 \nL 4684 1100 \nL 678 1100 \nL 678 1631 \nz\n\" id=\"DejaVuSans-3d\" transform=\"scale(0.015625)\"/>\n      </defs>\n      <use xlink:href=\"#DejaVuSans-54\"/>\n      <use x=\"44.583984\" xlink:href=\"#DejaVuSans-61\"/>\n      <use x=\"105.863281\" xlink:href=\"#DejaVuSans-6e\"/>\n      <use x=\"169.242188\" xlink:href=\"#DejaVuSans-67\"/>\n      <use x=\"232.71875\" xlink:href=\"#DejaVuSans-65\"/>\n      <use x=\"294.242188\" xlink:href=\"#DejaVuSans-6e\"/>\n      <use x=\"357.621094\" xlink:href=\"#DejaVuSans-74\"/>\n      <use x=\"396.830078\" xlink:href=\"#DejaVuSans-20\"/>\n      <use x=\"428.617188\" xlink:href=\"#DejaVuSans-6c\"/>\n      <use x=\"456.400391\" xlink:href=\"#DejaVuSans-69\"/>\n      <use x=\"484.183594\" xlink:href=\"#DejaVuSans-6e\"/>\n      <use x=\"547.5625\" xlink:href=\"#DejaVuSans-65\"/>\n      <use x=\"609.085938\" xlink:href=\"#DejaVuSans-20\"/>\n      <use x=\"640.873047\" xlink:href=\"#DejaVuSans-28\"/>\n      <use x=\"679.886719\" xlink:href=\"#DejaVuSans-78\"/>\n      <use x=\"739.066406\" xlink:href=\"#DejaVuSans-3d\"/>\n      <use x=\"822.855469\" xlink:href=\"#DejaVuSans-31\"/>\n      <use x=\"886.478516\" xlink:href=\"#DejaVuSans-29\"/>\n     </g>\n    </g>\n   </g>\n  </g>\n </g>\n <defs>\n  <clipPath id=\"p3ebdaefcff\">\n   <rect height=\"135.9\" width=\"195.3\" x=\"40.603125\" y=\"10.7\"/>\n  </clipPath>\n </defs>\n</svg>\n",
   "text/plain": "<Figure size 252x180 with 1 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 }
]
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
1. Given a function $f(x)$ that is invertible, compute the derivative of its inverse $f^{-1}(x)$. Here we have that $f^{-1}(f(x)) = x$ and conversely $f(f^{-1}(x)) =  

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
