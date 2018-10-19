# %%%1%%%

%%%2%%%

## %%%3%%%

%%%4%%%

%%%5%%%

%%%6%%%

%%%7%%%

%%%8%%%

%%%9%%%

%%%10%%%

%%%11%%%

%%%12%%%

%%%13%%%

%%%14%%%

```{.python .input  n=3}
%matplotlib inline
import gluonbook as gb
import math
from mxnet import nd
import numpy as np
```

%%%15%%%

```{.python .input  n=4}
def gd(eta):
    x = 10
    results = [x]
    for i in range(10):
        x -= eta * 2 * x  # %%%16%%%
        results.append(x)
    print('epoch 10, x:', x)
    return results

res = gd(0.2)
```

%%%17%%%

```{.python .input  n=5}
def show_trace(res):
    n = max(abs(min(res)), abs(max(res)), 10)
    f_line = np.arange(-n, n, 0.1)
    gb.set_figsize()
    gb.plt.plot(f_line, [x * x for x in f_line])
    gb.plt.plot(res, [x * x for x in res], '-o')
    gb.plt.xlabel('x')
    gb.plt.ylabel('f(x)')

show_trace(res)
```

## %%%18%%%

%%%19%%%

```{.python .input  n=6}
show_trace(gd(0.05))
```

%%%20%%%

```{.python .input  n=7}
show_trace(gd(1.1))
```

## %%%21%%%

%%%22%%%

%%%23%%%


为表示简洁，我们用$\nabla f(\boldsymbol{x})$代替$\nabla_{\boldsymbol{x}} f(\boldsymbol{x})$。梯度中每个偏导数元素$\partial f(\boldsymbol{x})/\partial x_i$代表着$f$在$\boldsymbol{x}$有关输入$x_i$的变化率。为了测量$f$沿着单位向量$\boldsymbol{u}$（即$\|\boldsymbol{u}\|=1$）方向上的变化率，在多元微积分中，我们定义$f$在$\boldsymbol{x}$上沿着$\boldsymbol{u}$方向的方向导数为

%%%25%%%

依据方向导数性质 \[1，14.6节定理三\]，以上的方向导数可以改写为

%%%27%%%

%%%28%%%

%%%29%%%

%%%30%%%

%%%31%%%

%%%32%%%

```{.python .input  n=10}
def train_2d(trainer):  # %%%33%%%
    x1, x2, s1, s2 = -5, -2, 0, 0  # %%%34%%%
    results = [(x1, x2)]
    for i in range(20):
        x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print('epoch %d, x1 %f, x2 %f' % (i + 1, x1, x2))
    return results

def show_trace_2d(f, results):  # %%%35%%%
    gb.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1))
    gb.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    gb.plt.xlabel('x1')
    gb.plt.ylabel('x2')
```

%%%36%%%

```{.python .input  n=15}
eta = 0.1

def f_2d(x1, x2):  # %%%37%%%
    return x1 ** 2 + 2 * x2 ** 2

def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 2 * x1, x2 - eta * 4 * x2, 0, 0)

show_trace_2d(f_2d, train_2d(gd_2d))
```

## %%%38%%%

%%%39%%%

%%%40%%%

%%%41%%%

%%%42%%%

%%%43%%%

随机梯度下降（stochastic gradient descent，简称SGD）减少了每次迭代的计算开销。在随机梯度下降的每次迭代中，我们随机均匀采样的一个样本索引$i\in\{1,\ldots,n\}$，并计算梯度$\nabla f_i(\boldsymbol{x})$来迭代$\boldsymbol{x}$：

%%%45%%%

%%%46%%%

%%%47%%%

%%%48%%%

%%%49%%%

```{.python .input  n=17}
def sgd_2d(x1, x2, s1, s2):
    return (x1 - eta * (2 * x1 + np.random.normal(0.1)),
            x2 - eta * (4 * x2 + np.random.normal(0.1)), 0, 0)

show_trace_2d(f_2d, train_2d(sgd_2d))
```

%%%50%%%


## %%%51%%%

* %%%52%%%
* %%%53%%%
* %%%54%%%


## %%%55%%%

* %%%56%%%
* %%%57%%%


## %%%58%%%

![](../img/qr_gd-sgd.svg)
