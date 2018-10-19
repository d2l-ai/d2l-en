# %%%1%%%

%%%2%%%

## %%%3%%%

%%%4%%%

%%%5%%% %%%6%%%

%%%7%%%

%%%8%%%

%%%9%%%

%%%10%%% %%%11%%%

%%%12%%%

%%%13%%% %%%14%%%

%%%15%%%


## %%%16%%%

%%%17%%%

```{.python .input  n=11}
%matplotlib inline
import gluonbook as gb
from mxnet import nd

features, labels = gb.get_data_ch7()

def init_adadelta_states():
    s_w, s_b = nd.zeros((features.shape[1], 1)), nd.zeros(1)
    delta_w, delta_b = nd.zeros((features.shape[1], 1)), nd.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        s[:] = rho * s + (1 - rho) * p.grad.square()
        g = ((delta + eps).sqrt() / (s + eps).sqrt()) * p.grad
        p[:] -= g
        delta[:] = rho * delta + (1 - rho) * g * g
```

%%%18%%%

```{.python .input  n=12}
gb.train_ch7(adadelta, init_adadelta_states(), {'rho': 0.9}, features, labels)
```

## %%%19%%%

%%%20%%%

```{.python .input  n=9}
gb.train_gluon_ch7('adadelta', {'rho': 0.9}, features, labels)
```

## %%%21%%%

* %%%22%%%

## %%%23%%%

* %%%24%%%

## %%%25%%%

![](../img/qr_adadelta.svg)

## %%%27%%%

%%%28%%% %%%29%%% %%%30%%% %%%31%%%
