# Nadaraya--Watson Kernel Regression
:label:`sec_nadaraya-waston`

```{.python .input  n=2}
from mxnet import np, npx

npx.set_np()

def gaussian(x):
    return np.exp(-np.power(x, 2) / 2)

```

## Nonparametric Model

The Nadaraya--Watson estimator is:

$$
f(x) = \sum_i \frac{K(x - x_i)}{\sum_j K(x - x_j)} y_i
$$

where Gaussian kernel is:

$$
K(u) = \frac{1}{\sqrt{2\pi}} \exp(-\frac{u^2}{2})
$$


Thus,

$$\begin{aligned} f(x) &= \sum_i \alpha(x, x_i) y_i \\&= \sum_i \frac{\exp\left(-\frac{1}{2}(x - x_i)^2\right)}{\sum_j \exp\left(-\frac{1}{2}(x - x_j)^2\right)} y_i \\&= \sum_i \mathrm{softmax}\left(-\frac{1}{2}(x - x_i)^2\right) y_i \end{aligned} $$


## Parametric Model

$$\begin{aligned}
f(x) &= \sum_i \alpha(x, x_i) y_i \\&= \sum_i \frac{\exp\left(-\frac{1}{2}\left(\frac{x - x_i}{w}\right)^2\right)}{\sum_j \exp\left(-\frac{1}{2}\left(\frac{x - x_j}{w}\right)^2\right)} y_i \\&= \sum_i \mathrm{softmax}\left(-\frac{1}{2}\left(\frac{x - x_i}{w}\right)^2\right) y_i
\end{aligned}$$
