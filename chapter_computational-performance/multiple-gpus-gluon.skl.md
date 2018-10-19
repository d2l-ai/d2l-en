# %%%1%%%

%%%2%%%

%%%3%%%

```{.python .input  n=1}
import gluonbook as gb
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn, utils as gutils
import time
```

## %%%4%%%

%%%5%%%

```{.python .input  n=2}
def resnet18(num_classes):  # %%%6%%%
    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(gb.Residual(
                    num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(gb.Residual(num_channels))
        return blk

    net = nn.Sequential()
    # %%%7%%%
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'))
    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net

net = resnet18(10)
```

%%%8%%%

```{.python .input  n=3}
ctx = [mx.gpu(0), mx.gpu(1)]
net.initialize(init=init.Normal(sigma=0.01), ctx=ctx)
```

%%%9%%%

```{.python .input  n=4}
x = nd.random.uniform(shape=(4, 1, 28, 28))
gpu_x = gutils.split_and_load(x, ctx)
net(gpu_x[0]), net(gpu_x[1])
```

%%%10%%%

```{.python .input  n=5}
weight = net[0].params.get('weight')

try:
    weight.data()
except RuntimeError:
    print('not initialized on', mx.cpu())
weight.data(ctx[0])[0], weight.data(ctx[1])[0]
```

## %%%11%%%

%%%12%%%

```{.python .input  n=7}
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)
    ctx = [mx.gpu(i) for i in range(num_gpus)]
    print('running on:', ctx)
    net.initialize(init=init.Normal(sigma=0.01), ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd', {'learning_rate': lr})
    loss = gloss.SoftmaxCrossEntropyLoss()
    for epoch in range(4):
        start = time.time()
        for X, y in train_iter:
            gpu_Xs = gutils.split_and_load(X, ctx)
            gpu_ys = gutils.split_and_load(y, ctx)
            with autograd.record():
                ls = [loss(net(gpu_X), gpu_y)
                      for gpu_X, gpu_y in zip(gpu_Xs, gpu_ys)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
        nd.waitall()
        train_time = time.time() - start
        test_acc = gb.evaluate_accuracy(test_iter, net, ctx[0])
        print('epoch %d, training time: %.1f sec, test_acc %.2f' % (
            epoch + 1, train_time, test_acc))
```

%%%13%%%

```{.python .input}
train(num_gpus=1, batch_size=256, lr=0.1)
```

%%%14%%%

```{.python .input  n=10}
train(num_gpus=2, batch_size=512, lr=0.2)
```

## %%%15%%%

* %%%16%%%

## %%%17%%%

* %%%18%%%
* %%%19%%%

## %%%20%%%

![](../img/qr_multiple-gpus-gluon.svg)
