# %%%1%%%

%%%2%%%

%%%3%%%
%%%4%%%

%%%5%%%


## %%%6%%%

%%%7%%%

%%%8%%%

%%%9%%%

%%%10%%%

%%%11%%%

## %%%12%%%

%%%13%%%

%%%14%%%

```{.python .input  n=2}
%matplotlib inline
import gluonbook as gb
import math
from mxnet import nd

def adagrad_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6  # %%%15%%%
    s1 += g1 ** 2
    s2 += g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta = 0.4
gb.show_trace_2d(f_2d, gb.train_2d(adagrad_2d))
```

%%%16%%%

```{.python .input  n=3}
eta = 2
gb.show_trace_2d(f_2d, gb.train_2d(adagrad_2d))
```

## %%%17%%%

%%%18%%%

```{.python .input  n=4}
features, labels = gb.get_data_ch7()

def init_adagrad_states():
    s_w = nd.zeros((features.shape[1], 1))
    s_b = nd.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        s[:] += p.grad.square()
        p[:] -= hyperparams['lr'] * p.grad / (s + eps).sqrt()
```

%%%19%%%

```{.python .input  n=5}
gb.train_ch7(adagrad, init_adagrad_states(), {'lr': 0.1}, features, labels)
```

## %%%20%%%

%%%21%%%

```{.python .input  n=6}
gb.train_gluon_ch7('adagrad', {'learning_rate': 0.1}, features, labels)
```

## %%%22%%%

* %%%23%%%
* %%%24%%%

## %%%25%%%

* %%%26%%%
* %%%27%%%


## %%%28%%%

![](../img/qr_adagrad.svg)


## %%%30%%%

%%%31%%% %%%32%%% %%%33%%% %%%34%%%
