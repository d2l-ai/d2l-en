# %%%1%%%

%%%2%%%


## %%%3%%%

%%%4%%%

```{.python .input  n=3}
%matplotlib inline
import gluonbook as gb
from mxnet import nd

eta = 0.4

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

gb.show_trace_2d(f_2d, gb.train_2d(gd_2d))
```

%%%5%%%

%%%6%%%

```{.python .input  n=4}
eta = 0.6
gb.show_trace_2d(f_2d, gb.train_2d(gd_2d))
```

## %%%7%%%

%%%8%%%

$$
\begin{aligned}
\boldsymbol{v}_t &\leftarrow \gamma \boldsymbol{v}_{t-1} + \eta_t \boldsymbol{g}_t, \\
\boldsymbol{x}_t &\leftarrow \boldsymbol{x}_{t-1} - \boldsymbol{v}_t,
\end{aligned}
$$

%%%10%%%

%%%11%%%

```{.python .input  n=5}
def momentum_2d(x1, x2, v1, v2):
    v1 = gamma * v1 + eta * 0.2 * x1
    v2 = gamma * v2 + eta * 4 * x2
    return x1 - v1, x2 - v2, v1, v2

eta, gamma = 0.4, 0.5
gb.show_trace_2d(f_2d, gb.train_2d(momentum_2d))
```

%%%12%%%

```{.python .input  n=11}
eta = 0.6
gb.show_trace_2d(f_2d, gb.train_2d(momentum_2d))
```

### %%%13%%%

%%%14%%%

%%%15%%%

%%%16%%%

$$
\begin{aligned}
y_t  &= (1-\gamma) x_t + \gamma y_{t-1}\\
         &= (1-\gamma)x_t + (1-\gamma) \cdot \gamma x_{t-1} + \gamma^2y_{t-2}\\
         &= (1-\gamma)x_t + (1-\gamma) \cdot \gamma x_{t-1} + (1-\gamma) \cdot \gamma^2x_{t-2} + \gamma^3y_{t-3}\\
         &\ldots
\end{aligned}
$$

%%%18%%%

%%%19%%%

%%%20%%%

%%%21%%%

%%%22%%%


### %%%23%%%

%%%24%%%

%%%25%%% %%%26%%%

由指数加权移动平均的形式可得，速度变量$\boldsymbol{v}_t$实际上对序列$\{\eta_{t-i}\boldsymbol{g}_{t-i} /(1-\gamma):i=0,\ldots,1/(1-\gamma)-1\}$做了指数加权移动平均。换句话说，相比于小批量随机梯度下降，动量法在每个时间步的自变量更新量近似于将前者对应的最近$1/(1-\gamma)$个时间步的更新量做了指数加权移动平均后再除以$1-\gamma$。所以动量法中，自变量在各个方向上的移动幅度不仅取决当前梯度，还取决于过去的各个梯度在各个方向上是否一致。在本节之前示例的优化问题中，所有梯度在水平方向上为正（向右）、而在竖直方向上时正（向上）时负（向下）。这样，我们就可以使用较大的学习率，从而使自变量向最优解更快移动。


## %%%28%%%

%%%29%%%

```{.python .input  n=13}
features, labels = gb.get_data_ch7()

def init_momentum_states():
    v_w = nd.zeros((features.shape[1], 1))
    v_b = nd.zeros(1)
    return (v_w, v_b)

def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        v[:] = hyperparams['momentum'] * v + hyperparams['lr'] * p.grad
        p[:] -= v
```

%%%30%%%

```{.python .input  n=15}
gb.train_ch7(sgd_momentum, init_momentum_states(),
             {'lr': 0.02, 'momentum': 0.5}, features, labels)
```

%%%31%%%

```{.python .input  n=8}
gb.train_ch7(sgd_momentum, init_momentum_states(),
             {'lr': 0.02, 'momentum': 0.9}, features, labels)
```

%%%32%%%

```{.python .input}
gb.train_ch7(sgd_momentum, init_momentum_states(),
             {'lr': 0.004, 'momentum': 0.9}, features, labels)
```

## %%%33%%%

%%%34%%%

```{.python .input  n=9}
gb.train_gluon_ch7('sgd', {'learning_rate': 0.004, 'momentum': 0.9}, features,
                   labels)
```

## %%%35%%%

* %%%36%%%
* %%%37%%%

## %%%38%%%

* %%%39%%%


## %%%40%%%

![](../img/qr_momentum.svg)
