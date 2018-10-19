# %%%1%%%

%%%2%%%

```{.python .input}
def add(a, b):
    return a + b

def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g

fancy_func(1, 2, 3, 4)
```

%%%3%%%

%%%4%%%

%%%5%%%

1. %%%6%%%
2. %%%7%%%
3. %%%8%%%

%%%9%%%

```{.python .input}
def add_str():
    return '''
def add(a, b):
    return a + b
'''

def fancy_func_str():
    return '''
def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g
'''

def evoke_str():
    return add_str() + fancy_func_str() + '''
print(fancy_func(1, 2, 3, 4))
'''

prog = evoke_str()
print(prog)
y = compile(prog, '', 'exec')
exec(y)
```

%%%10%%%

%%%11%%%

* %%%12%%%

* %%%13%%%


## %%%14%%%

%%%15%%%

%%%16%%%

%%%17%%%

## %%%18%%%

%%%19%%%

```{.python .input}
from mxnet import nd, sym
from mxnet.gluon import nn
import time

def get_net():
    net = nn.HybridSequential()  # %%%20%%%
    net.add(nn.Dense(256, activation='relu'),
            nn.Dense(128, activation='relu'),
            nn.Dense(2))
    net.initialize()
    return net

x = nd.random.normal(shape=(1, 512))
net = get_net()
net(x)
```

%%%21%%%

```{.python .input}
net.hybridize()
net(x)
```

%%%22%%%


### %%%23%%%

%%%24%%%

```{.python .input}
def benchmark(net, x):
    start = time.time()
    for i in range(1000):
        _ = net(x)
    nd.waitall()  # %%%25%%%
    return time.time() - start

net = get_net()
print('before hybridizing: %.4f sec' % (benchmark(net, x)))
net.hybridize()
print('after hybridizing: %.4f sec' % (benchmark(net, x)))
```

%%%26%%%


### %%%27%%%

%%%28%%%

```{.python .input}
net.export('my_mlp')
```

%%%29%%%

%%%30%%%

```{.python .input}
x = sym.var('data')
net(x)
```

## %%%31%%%

%%%32%%%

%%%33%%%

```{.python .input}
class HybridNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(HybridNet, self).__init__(**kwargs)
        self.hidden = nn.Dense(10)
        self.output = nn.Dense(2)

    def hybrid_forward(self, F, x):
        print('F: ', F)
        print('x: ', x)
        x = F.relu(self.hidden(x))
        print('hidden: ', x)
        return self.output(x)
```

%%%34%%%

%%%35%%%

```{.python .input}
net = HybridNet()
net.initialize()
x = nd.random.normal(shape=(1, 4))
net(x)
```

%%%36%%%

```{.python .input}
net(x)
```

%%%37%%%

```{.python .input}
net.hybridize()
net(x)
```

%%%38%%%

%%%39%%%

```{.python .input}
net(x)
```

%%%40%%%


## %%%41%%%

* %%%42%%%
* %%%43%%%


## %%%44%%%

* %%%45%%%
* %%%46%%%
* %%%47%%%


## %%%48%%%

![](../img/qr_hybridize.svg)
