# %%%1%%%

%%%2%%%

## %%%3%%%

%%%4%%%

%%%5%%% %%%6%%%

%%%7%%%

%%%8%%%

%%%9%%%

%%%10%%%

```{.python .input  n=3}
%matplotlib inline
import gluonbook as gb
import math
from mxnet import nd

def rmsprop_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    s1 = gamma * s1 + (1 - gamma) * g1 ** 2
    s2 = gamma * s2 + (1 - gamma) * g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta, gamma = 0.4, 0.9
gb.show_trace_2d(f_2d, gb.train_2d(rmsprop_2d))
```

## %%%11%%%

%%%12%%%

```{.python .input  n=22}
features, labels = gb.get_data_ch7()

def init_rmsprop_states():
    s_w = nd.zeros((features.shape[1], 1))
    s_b = nd.zeros(1)
    return (s_w, s_b)

def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        s[:] = gamma * s + (1 - gamma) * p.grad.square()
        p[:] -= hyperparams['lr'] * p.grad / (s + eps).sqrt()
```

%%%13%%%

```{.python .input  n=24}
features, labels = gb.get_data_ch7()
gb.train_ch7(rmsprop, init_rmsprop_states(), {'lr': 0.01, 'gamma': 0.9},
             features, labels)
```

## %%%14%%%

%%%15%%%

```{.python .input  n=29}
gb.train_gluon_ch7('rmsprop', {'learning_rate': 0.01, 'gamma1': 0.9},
                   features, labels)
```

## %%%16%%%

* %%%17%%%

## %%%18%%%

* %%%19%%%
* %%%20%%%

## %%%21%%%


![](../img/qr_rmsprop.svg)

## %%%23%%%

%%%24%%% %%%25%%% %%%26%%% %%%27%%%
