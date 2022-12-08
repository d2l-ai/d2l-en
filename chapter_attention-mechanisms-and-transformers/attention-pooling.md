# Attention Pooling by Similarity

:label:`sec_attention-pooling`

Now that we introduced the primary components of the attention mechanism, let's use them in a rather classical setting, namely regression and classification using kernel density estimation :cite:`Nadaraya.1964,Watson.1964`. This detour is entirely optional and can be skipped if needed. It simply provides additional background.
At their core, Nadaraya-Watson estimators rely on some similarity kernel $\alpha(\mathbf{k}, \mathbf{q})$ relating queries $\mathbf{q}$ to keys $\mathbf{k}$. Some common kernels are

$$\begin{aligned}
\alpha(\mathbf{k}, \mathbf{q}) & = \exp\left(\frac{1}{2} \|\mathbf{k} - \mathbf{q}\|^2 \right) && \mathrm{Gaussian} \\
\alpha(\mathbf{k}, \mathbf{q}) & = 1 \text{ if } \|\mathbf{k} - \mathbf{q}\| \leq 1 && \mathrm{Boxcar} \\
\alpha(\mathbf{k}, \mathbf{q}) & = \mathop{\mathrm{max}}\left(0, 1 - \|\mathbf{k} - \mathbf{q}\|\right) && \mathrm{Epanechikov}
\end{aligned}
$$

There are many more choices that we could pick. This [Wikipedia article](https://en.wikipedia.org/wiki/Kernel_(statistics)) has a more extensive review and shows how the choice of kernels is related to kernel density estimation, somtimes also called Parzen Windows :cite:`parzen1957consistent`. All of the kernels are heuristic and can be tuned. For instance, we can adjust the width, not only on a global basis but even on a per-coordinate basis. Regardless, all of them lead to the following equation for regresion and classification alike:

$$f(\mathbf{q}) = \sum_i \mathbf{v}_i \frac{\alpha(\mathbf{k}_i, \mathbf{q})}{\sum_j \alpha(\mathbf{k}, \mathbf{q}_j)}$$

In the case of a (scalar) regression with observations $(\mathbf{x}_i, y_i)$ for covariates and labels respectively, $\mathbf{v}_i = y_i$ are scalars, $\mathbf{k}_i = \mathbf{x}_i$ are vectors, and the query $\mathbf{q}$ denotes the new location where $f$ should be evaluated. In the case of (multiclass) classification, we use one-hot-encoding of $y_i$ to obtain $\mathbf{v}_i$. One of the convenient properties of this estimator is that it requires no training. Even more so, if we suitably narrow the kernel with increasing amounts of data, the approach is consistent :cite:`mack1982weak`, i.e., it will converge to some statistically optimal solution. Let's start by inspecting some kernels.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow')
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import numpy as np
```

## [**Kernels and Data**]

All the kernels $\alpha(\mathbf{k}, \mathbf{q})$ defined in this section are *translation and rotation invariant*, that is, if we shift and rotate $\mathbf{k}$ and $\mathbf{q}$ in the same manner, the value of $\alpha$ remains unchanged. For simplicity we thus pick scalar arguments $\mathbf{k}, \mathbf{q} \in \mathbb{R}$ and pick the key $k = 0$ as the origin. This yields:

```{.python .input}
%%tab all
d2l.use_svg_display()
fig, axes = d2l.plt.subplots(1, 4, sharey=True, figsize = (12,3))

# define some kernels
gauss = lambda x: d2l.exp(-2 * x**2)
boxcar = lambda x: d2l.abs(x) < 1.0
constant = lambda x: 1.0 + 0 * x
if tab.selected('pytorch'):
    epanechikov = lambda x: torch.max(1 - d2l.abs(x), torch.zeros_like(x))
if tab.selected('mxnet'):
    epanechikov = lambda x: np.maximum(1 - d2l.abs(x), 0)
if tab.selected('tensorflow'):
    epanechikov = lambda x: tf.math.maximum(1 - d2l.abs(x), 0)
kernels = (gauss, boxcar, constant, epanechikov)
names = ('Gauss', 'Boxcar', 'Constant', 'Epanechikov')

x = d2l.arange(-2.5, 2.5, 0.1)
for kernel, name, ax in zip(kernels, names, axes):
    ax.plot(d2l.numpy(x), d2l.numpy(kernel(x)));
    ax.set_xlabel(name)
```

Different kernels correspond to different notions of range and smoothness respectively. For instance, the boxcar kernel only attends to observations within a distance of $1$ (or some otherwise defined hyperparameter) and does so indiscriminately. 

To see Watson-Nadaraya estimation in action, let's define some training data. In the following we use the dependency

$$y_i = 2\sin(x_i) + x_i + \epsilon,$$

where $\epsilon$ is drawn from a Normal Distribution with zero mean and unit variance. We draw 50 samples.

```{.python .input}
%%tab all
n = 40
f = lambda x: 2 * d2l.sin(x) + x
if tab.selected('pytorch'):
    x_train, _ = torch.sort(d2l.rand(n) * 5)
    y_train = f(x_train) + d2l.randn(n)
if tab.selected('mxnet'):
    x_train = np.sort(d2l.rand(n) * 5)
    y_train = f(x_train) + d2l.randn(n)
if tab.selected('tensorflow'):
    x_train = tf.sort(d2l.rand((n,1)) * 5, 0)
    y_train = f(x_train) + d2l.normal((n,1))
x_test = d2l.arange(0, 5, 0.1)
y_test = f(x_test)
```

## [**Nadaraya-Watson Regression**]

Now that we have data and kernels, all we need is a function that computes the kernel regression estimates. Note that we also want to obtain the relative kernel weights, aka, the attention weights, in order to perform some minor diagnostics. Hence we first compute the kernel between all training covariates `x_train` and all test covariates `x_test`. This yields a matrix, which we subsequently normalize. When multiplied with the training labels `y_train` we obtain the estimates.

```{.python .input}
%%tab all
def watson_nadaraya(x_train, y_train, x_test, kernel):
    dists = d2l.reshape(x_train, (-1, 1)) - d2l.reshape(x_test, (1, -1))
    k = kernel(dists) # compute attention mask
    k = k / k.sum(0)  # normalization
    if tab.selected('pytorch'):
        y_hat = y_train@k
    if tab.selected('mxnet'):
        y_hat = np.dot(y_train, k)
    if tab.selected('tensorflow'):
        y_hat = tf.matmul(y_train, k)
    return y_hat, k
```

Let's have a look at the kind of estimates that the different kernels produce.

```{.python .input}
def plot_results(x_train, y_train, x_test, y_test, kernels, names, attention = False):
    fig, axes = d2l.plt.subplots(1, 4, sharey=True, figsize = (12, 3))
    for kernel, name, ax in zip(kernels, names, axes):
        y_hat, k = watson_nadaraya(x_train, y_train, x_test, kernel)
        if attention: pcm = ax.imshow(d2l.numpy(k), cmap='Reds')
        else:
            ax.plot(x_test, y_hat)
            ax.plot(x_test, y_test)
            ax.plot(x_train, y_train, 'o', alpha=0.5);
        ax.set_xlabel(name)
    if attention: fig.colorbar(pcm, ax=axes, shrink=0.7)
        
plot_results(x_train, y_train, x_test, y_test, kernels, names)
```

The first thing that stands out is that all three nontrivial kernels (Gauss, Boxcar and Epanechikov) produce fairly workable estimates that are not too far from the true function. Only the constant kernel which leads to the trivial estimate $f(x) = \frac{1}{n} \sum_i y_i$ produces a rather unrealistic result. Let's inspect the attention weighting a bit more closely:

```{.python .input}
plot_results(x_train, y_train, x_test, y_test, kernels, names, attention = True)
```

The visualization clearly shows why the estimages for Gauss, Boxcar and Epanechikov are very similar - after all, they are derived from very similar attention weights, despite the different functional form of the kernel. This raises the question as to whether this is always the case. 

## [**Adapting Attention Pooling**]

We could replace the (Gaussian) kernel with one of a different width. That is, we could use 
$\alpha(\mathbf{k}, \mathbf{q}) = \exp\left(\frac{1}{2 \sigma^2} \|\mathbf{k} - \mathbf{q}\|^2 \right)$ where $\sigma^2$ determines the width of the kernel. Let's see whether this affects the outcomes.

```{.python .input}
sigmas = (0.1, 0.2, 0.5, 1)
names = ['Sigma ' + str(sigma) for sigma in sigmas]
def getfun(sigma): return (lambda x: d2l.exp(-(1/(2*sigma**2)) * x**2))
kernels = [getfun(sigma) for sigma in sigmas]

plot_results(x_train, y_train, x_test, y_test, kernels, names)
```

Clearly, the narrower the kernel, the less smooth the estimate. At the same time, it adapts better to the local variations. Let's look at the corresponding attention weights.

```{.python .input}
plot_results(x_train, y_train, x_test, y_test, kernels, names, attention=True)
```

As we would expect, the narrower the kernel, the narrower the range of large attention weights. It's also clear that picking the same width might not be ideal. In fact, :cite:`Silverman86` proposed a heuristic that depends on the local density. Many more such 'tricks' have been proposed. It remains a valuable technique to date. For instance, :cite:`norelli2022asif` use a similar nearest-neighbor interpolation technique to design cross-modal image and text representations. 

The astute reader might wonder why this deep-dive on a method that is over half a century old: firstly, it is one of the earliest precursors of modern attention mechanisms. Secondly, it is great for visualization. Third, and just as importantly, it demonstrates the limits of hand-crafted attention mechanisms. A much better strategy is to *learn* the mechanism, by learning the representations for queries and keys. This what we will embark on in the following sections.

## Summary

Nadaraya-Watson kernel regression is an early precursor of the current attention mechanisms. 
It can be used directly with little to no training or tuning, both for classification and regression. 
The attention weight is assigned according to the similarity (read, distance) between query and key and according to how many similar observations are available. 

## Exercises

1. Parzen windows density estimates are given by $\hat{p}(\mathbf{x}) = \frac{1}{n} \sum_i k(\mathbf{x}, \mathbf{x}_i)$. Prove that for binary classification the function $\hat{p}(\mathbf{x}, y=1) - \hat{p}(\mathbf{x}, y=-1)$, as obtained by Parzen windows is equivalent to Nadaraya-Watson classification. 
1. Implement SGD to learn a good value for kernels widths in Watson Nadarya regression. 
    1. What happens if you just use the above estimates to minimize $(f(\mathbf{x_i}) - y_i)^2$ directly? Hint: y_i is part of the terms used to compute $f$.
    1. Remove $(\mathbf{x}_i, y_i)$ from the estimate for $f(\mathbf{x_i})$ and optimize over the kernel widths. Do you still observe overfitting?
1. Assume that all $\mathbf{x}$ lie on the unit sphere, i.e., all satisfy $\|\mathbf{x}\| = 1$. Can you simplify the $\|\mathbf{x}_i - \mathbf{x}\|^2$ term in the exponential? Hint: we will later see that this is very closely related to dot-product attention. 
1. Recall that :cite:`mack1982weak` prove that Watson-Nadaraya estimation is consistent. How quickly should you reduce the scale for the attention mechanism as you get more data? Provide some intuition for your answer. Does it depend on the dimensionality of the data? How?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1598)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1599)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/3866)
:end_tab:
