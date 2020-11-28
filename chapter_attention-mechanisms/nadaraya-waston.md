# Nadaraya--Watson Kernel Regression
:label:`sec_nadaraya-waston`

```{.python .input  n=2}
from d2l import mxnet as d2l
from mxnet import np, npx


npx.set_np()

def gaussian(x):
    return d2l.exp(-np.power(x, 2) / 2)


def f(x):
    return 2 * d2l.sin(x) + x**0.8
```

```{.python .input}
n_train = 50
x_train = np.random.rand(n_train) * 5
y_train = f(x_train) + d2l.normal(0, 0.5, n_train)

x = np.arange(0, 5, 0.05)
y_truth = f(x)

def plot_kernel_reg(y_pred):
    d2l.plot(x, [y_truth, y_pred], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5);
```

## Average Pooling

```{.python .input}
y_pred = np.repeat(y_train.mean(), len(x))
plot_kernel_reg(y_pred)
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

```{.python .input}
X_repeat = d2l.reshape(np.repeat(x, n_train), (-1, n_train))
attention_matrix = npx.softmax(-(X_repeat - x_train)**2 / 2)
y_pred = d2l.matmul(attention_matrix, y_train)
plot_kernel_reg(y_pred)
```




## Parametric Model

$$\begin{aligned}
f(x) &= \sum_i \alpha(x, x_i) y_i \\&= \sum_i \frac{\exp\left(-\frac{1}{2}((x - x_i)w)^2\right)}{\sum_j \exp\left(-\frac{1}{2}((x - x_i)w)^2\right)} y_i \\&= \sum_i \mathrm{softmax}\left(-\frac{1}{2}((x - x_i)w)^2\right) y_i
\end{aligned}$$
