# %%%1%%%

%%%2%%%

```{.python .input  n=1}
from mxnet import autograd, gluon, nd
from mxnet.gluon import loss as gloss, nn
import os
import subprocess
import time
```

## %%%3%%%

%%%4%%%

%%%5%%%

```{.python .input  n=3}
a = nd.ones((1, 2))
b = nd.ones((1, 2))
c = a * b + 2
c
```

%%%6%%%

%%%7%%%

```{.python .input}
class Benchmark():  # %%%8%%%
    def __init__(self, prefix=None):
        self.prefix = prefix + ' ' if prefix else ''

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        print('%stime: %.4f sec' % (self.prefix, time.time() - self.start))
```

%%%9%%%

```{.python .input  n=4}
with Benchmark('Workloads are queued.'):
    x = nd.random.uniform(shape=(2000, 2000))
    y = nd.dot(x, x).sum()

with Benchmark('Workloads are finished.'):
    print('sum =', y)
```

%%%10%%%


## %%%11%%%

%%%12%%%

%%%13%%%

```{.python .input  n=5}
with Benchmark():
    y = nd.dot(x, x)
    y.wait_to_read()
```

%%%14%%%

```{.python .input  n=6}
with Benchmark():
    y = nd.dot(x, x)
    z = nd.dot(x, x)
    nd.waitall()
```

%%%15%%%

```{.python .input  n=7}
with Benchmark():
    y = nd.dot(x, x)
    y.asnumpy()
```

```{.python .input  n=8}
with Benchmark():
    y = nd.dot(x, x)
    y.norm().asscalar()
```

%%%16%%%


## %%%17%%%

%%%18%%%

```{.python .input  n=9}
with Benchmark('synchronous.'):
    for _ in range(1000):
        y = x + 1
        y.wait_to_read()

with Benchmark('asynchronous.'):
    for _ in range(1000):
        y = x + 1
    nd.waitall()
```

%%%19%%%

1. %%%20%%%
1. %%%21%%%
1. %%%22%%%

%%%23%%%

## %%%24%%%

%%%25%%%

%%%26%%%

%%%27%%%

```{.python .input  n=11}
def data_iter():
    start = time.time()
    num_batches, batch_size = 100, 1024
    for i in range(num_batches):
        X = nd.random.normal(shape=(batch_size, 512))
        y = nd.ones((batch_size,))
        yield X, y
        if (i + 1) % 50 == 0:
            print('batch %d, time %f sec' % (i + 1, time.time() - start))
```

%%%28%%%

```{.python .input  n=12}
net = nn.Sequential()
net.add(nn.Dense(2048, activation='relu'),
        nn.Dense(512, activation='relu'),
        nn.Dense(1))
net.initialize()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.005})
loss = gloss.L2Loss()
```

%%%29%%%

```{.python .input  n=13}
def get_mem():
    res = subprocess.check_output(['ps', 'u', '-p', str(os.getpid())])
    return int(str(res).split()[15]) / 1e3
```

%%%30%%%

```{.python .input  n=14}
for X, y in data_iter():
    break
loss(y, net(X)).wait_to_read()
```

%%%31%%%

```{.python .input  n=17}
l_sum, mem = 0, get_mem()
for X, y in data_iter():
    with autograd.record():
        l = loss(y, net(X))
    l_sum += l.mean().asscalar()  # %%%32%%%
    l.backward()
    trainer.step(X.shape[0])
nd.waitall()
print('increased memory: %f MB' % (get_mem() - mem))
```

%%%33%%%

```{.python .input  n=18}
mem = get_mem()
for X, y in data_iter():
    with autograd.record():
        l = loss(y, net(X))
    l.backward()
    trainer.step(x.shape[0])
nd.waitall()
print('increased memory: %f MB' % (get_mem() - mem))
```

## %%%34%%%

* %%%35%%%

* %%%36%%%

* %%%37%%%


## %%%38%%%

* %%%39%%%


## %%%40%%%

![](../img/qr_async-computation.svg)
