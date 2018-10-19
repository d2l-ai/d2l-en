# %%%1%%%

%%%2%%%

## %%%3%%%

%%%4%%%

%%%5%%% %%%6%%%

%%%7%%%

%%%8%%% %%%9%%%

%%%10%%%

%%%11%%%

%%%12%%% %%%13%%%


%%%14%%%

%%%15%%%

%%%16%%%

%%%17%%% %%%18%%%

## %%%19%%%

%%%20%%%

```{.python .input  n=2}
%matplotlib inline
import gluonbook as gb
from mxnet import nd

features, labels = gb.get_data_ch7()

def init_adam_states():
    v_w, v_b = nd.zeros((features.shape[1], 1)), nd.zeros(1)
    s_w, s_b = nd.zeros((features.shape[1], 1)), nd.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = beta2 * s + (1 - beta2) * p.grad.square()
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:] -= hyperparams['lr'] * v_bias_corr / (s_bias_corr.sqrt() + eps)
    hyperparams['t'] += 1
```

%%%21%%%

```{.python .input  n=5}
gb.train_ch7(adam, init_adam_states(), {'lr': 0.01, 't': 1}, features, labels)
```

## %%%22%%%

%%%23%%%

```{.python .input  n=11}
gb.train_gluon_ch7('adam', {'learning_rate': 0.01}, features, labels)
```

## %%%24%%%

* %%%25%%%
* %%%26%%%

## %%%27%%%

* %%%28%%%
* %%%29%%%


## %%%30%%%

![](../img/qr_adam.svg)

## %%%32%%%

%%%33%%% %%%34%%% %%%35%%% %%%36%%%
