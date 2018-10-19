# %%%1%%%

%%%2%%%

```{.python .input  n=1}
!nvidia-smi
```

%%%3%%%


## %%%4%%%

%%%5%%%

%%%6%%%

%%%7%%%

%%%8%%%

```{.python .input  n=2}
import gluonbook as gb
import mxnet as mx
from mxnet import autograd, nd
from mxnet.gluon import loss as gloss
import time
```

## %%%9%%%

%%%10%%%

```{.python .input  n=3}
# %%%11%%%
scale = 0.01
W1 = nd.random.normal(scale=scale, shape=(20, 1, 3, 3))
b1 = nd.zeros(shape=20)
W2 = nd.random.normal(scale=scale, shape=(50, 20, 5, 5))
b2 = nd.zeros(shape=50)
W3 = nd.random.normal(scale=scale, shape=(800, 128))
b3 = nd.zeros(shape=128)
W4 = nd.random.normal(scale=scale, shape=(128, 10))
b4 = nd.zeros(shape=10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]

# %%%12%%%
def lenet(X, params):
    h1_conv = nd.Convolution(data=X, weight=params[0], bias=params[1],
                             kernel=(3, 3), num_filter=20)
    h1_activation = nd.relu(h1_conv)
    h1 = nd.Pooling(data=h1_activation, pool_type='avg', kernel=(2, 2),
                    stride=(2, 2))
    h2_conv = nd.Convolution(data=h1, weight=params[2], bias=params[3],
                             kernel=(5, 5), num_filter=50)
    h2_activation = nd.relu(h2_conv)
    h2 = nd.Pooling(data=h2_activation, pool_type='avg', kernel=(2, 2),
                    stride=(2, 2))
    h2 = nd.flatten(h2)
    h3_linear = nd.dot(h2, params[4]) + params[5]
    h3 = nd.relu(h3_linear)
    y_hat = nd.dot(h3, params[6]) + params[7]
    return y_hat

# %%%13%%%
loss = gloss.SoftmaxCrossEntropyLoss()
```

## %%%14%%%

%%%15%%%

```{.python .input  n=4}
def get_params(params, ctx):
    new_params = [p.copyto(ctx) for p in params]
    for p in new_params:
        p.attach_grad()
    return new_params
```

%%%16%%%

```{.python .input  n=5}
new_params = get_params(params, mx.gpu(0))
print('b1 weight:', new_params[1])
print('b1 grad:', new_params[1].grad)
```

%%%17%%%

```{.python .input  n=6}
def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].copyto(data[0].context)
    for i in range(1, len(data)):
        data[0].copyto(data[i])
```

%%%18%%%

```{.python .input  n=7}
data = [nd.ones((1, 2), ctx=mx.gpu(i)) * (i + 1) for i in range(2)]
print('before allreduce:', data)
allreduce(data)
print('after allreduce:', data)
```

%%%19%%%

```{.python .input  n=8}
def split_and_load(data, ctx):
    n, k = data.shape[0], len(ctx)
    m = n // k  # %%%20%%%
    assert m * k == n, '# %%%21%%%
    return [data[i * m: (i + 1) * m].as_in_context(ctx[i]) for i in range(k)]
```

%%%22%%%

```{.python .input  n=9}
batch = nd.arange(24).reshape((6, 4))
ctx = [mx.gpu(0), mx.gpu(1)]
splitted = split_and_load(batch, ctx)
print('input: ', batch)
print('load into', ctx)
print('output:', splitted)
```

## %%%23%%%

%%%24%%%

```{.python .input  n=10}
def train_batch(X, y, gpu_params, ctx, lr):
    # %%%25%%%
    gpu_Xs, gpu_ys = split_and_load(X, ctx), split_and_load(y, ctx) 
    with autograd.record():  # %%%26%%%
        ls = [loss(lenet(gpu_X, gpu_W), gpu_y)
              for gpu_X, gpu_y, gpu_W in zip(gpu_Xs, gpu_ys, gpu_params)]
    for l in ls:  # %%%27%%%
        l.backward()
    # %%%28%%%
    for i in range(len(gpu_params[0])):
        allreduce([gpu_params[c][i].grad for c in range(len(ctx))])
    for param in gpu_params:  # %%%29%%%
        gb.sgd(param, lr, X.shape[0])  # %%%30%%%
```

## %%%31%%%

%%%32%%%

```{.python .input  n=11}
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)
    ctx = [mx.gpu(i) for i in range(num_gpus)]
    print('running on:', ctx)
    # %%%33%%%
    gpu_params = [get_params(params, c) for c in ctx]
    for epoch in range(4):
        start = time.time()
        for X, y in train_iter:
            # %%%34%%%
            train_batch(X, y, gpu_params, ctx, lr)
            nd.waitall()
        train_time = time.time() - start

        def net(x):  # %%%35%%%
            return lenet(x, gpu_params[0])

        test_acc = gb.evaluate_accuracy(test_iter, net, ctx[0])
        print('epoch %d, time: %.1f sec, test acc: %.2f'
              % (epoch + 1, train_time, test_acc))
```

## %%%36%%%

%%%37%%%

```{.python .input  n=12}
train(num_gpus=1, batch_size=256, lr=0.2)
```

%%%38%%%

```{.python .input  n=13}
train(num_gpus=2, batch_size=256, lr=0.2)
```

## %%%39%%%

* %%%40%%%
* %%%41%%%

## %%%42%%%

* %%%43%%%
* %%%44%%%


## %%%45%%%

![](../img/qr_multiple-gpus.svg)
