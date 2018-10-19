# %%%1%%%

%%%2%%%


%%%3%%%

%%%4%%%

%%%5%%%

%%%6%%%

%%%7%%%


%%%8%%%


## %%%9%%%

%%%10%%%

```{.python .input  n=1}
%matplotlib inline
import gluonbook as gb
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import nn, data as gdata, loss as gloss
import numpy as np
import time

def get_data_ch7():  # %%%11%%%
    data = np.genfromtxt('../data/airfoil_self_noise.dat', delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return nd.array(data[:1500, :-1]), nd.array(data[:1500, -1])

features, labels = get_data_ch7()
features.shape
```

## %%%12%%%

%%%13%%%

```{.python .input  n=3}
def sgd(params, states, hyperparams):
    for p in params:
        p[:] -= hyperparams['lr'] * p.grad
```

%%%14%%%

```{.python .input  n=4}
# %%%15%%%
def train_ch7(trainer_fn, states, hyperparams, features, labels,
              batch_size=10, num_epochs=2):
    # %%%16%%%
    net, loss = gb.linreg, gb.squared_loss
    w = nd.random.normal(scale=0.01, shape=(features.shape[1], 1))
    b = nd.zeros(1)
    w.attach_grad()
    b.attach_grad()

    def eval_loss():
        return loss(net(features, w, b), labels).mean().asscalar()

    ls = [eval_loss()]
    data_iter = gdata.DataLoader(
        gdata.ArrayDataset(features, labels), batch_size, shuffle=True)
    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            with autograd.record():
                l = loss(net(X, w, b), y).mean()  # %%%17%%%
            l.backward()
            trainer_fn([w, b], states, hyperparams)  # %%%18%%%
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())  # %%%19%%%
    # %%%20%%%
    print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
    gb.set_figsize()
    gb.plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    gb.plt.xlabel('epoch')
    gb.plt.ylabel('loss')
```

%%%21%%%

```{.python .input  n=5}
def train_sgd(lr, batch_size, num_epochs=2):
    train_ch7(sgd, None, {'lr': lr}, features, labels, batch_size, num_epochs)

train_sgd(1, 1500, 6)
```

%%%22%%%

%%%23%%%

```{.python .input  n=6}
train_sgd(0.005, 1)
```

%%%24%%%

```{.python .input  n=7}
train_sgd(0.05, 10)
```

## %%%25%%%

%%%26%%%

```{.python .input  n=8}
# %%%27%%%
def train_gluon_ch7(trainer_name, trainer_hyperparams, features, labels,
                    batch_size=10, num_epochs=2):
    # %%%28%%%
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))
    loss = gloss.L2Loss()

    def eval_loss():
        return loss(net(features), labels).mean().asscalar()

    ls = [eval_loss()]
    data_iter = gdata.DataLoader(
        gdata.ArrayDataset(features, labels), batch_size, shuffle=True)
    # %%%29%%%
    trainer = gluon.Trainer(
        net.collect_params(), trainer_name, trainer_hyperparams)
    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)  # %%%30%%%
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())
    # %%%31%%%
    print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
    gb.set_figsize()
    gb.plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    gb.plt.xlabel('epoch')
    gb.plt.ylabel('loss')
```

%%%32%%%

```{.python .input  n=9}
train_gluon_ch7('sgd', {'learning_rate': 0.05}, features, labels, 10)
```

## %%%33%%%

* %%%34%%%
* %%%35%%%
* %%%36%%%

## %%%37%%%

* %%%38%%%
* %%%39%%%


## %%%40%%%

![](../img/qr_minibatch-sgd.svg)

## %%%42%%%

%%%43%%%archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise
