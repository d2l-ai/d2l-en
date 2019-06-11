# Concise Implementation of Multi-GPU Computation
:label:`chapter_multi_gpu_gluon`

In Gluon, we can conveniently use data parallelism to perform multi-GPU
computation. For example, we do not need to implement the helper function to
synchronize data among multiple GPUs, as described in
:numref:`chapter_multi_gpu`, ourselves.

First, import the required packages or modules for the experiment in this section. Running the programs in this section requires at least two GPUs.

```{.python .input  n=1}
import d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import nn
```

## Initialize Model Parameters on Multiple GPUs

In this section, we use ResNet-18 as a sample model. Since the input images in
this section are original size (not enlarged), the model construction here is
different from the ResNet-18 structure described in :numref:`chapter_resnet`. This
model uses a smaller convolution kernel, stride, and padding at the beginning
and removes the maximum pooling layer.

```{.python .input  n=2}
# Save to the d2l package.
def resnet18(num_classes):
    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(d2l.Residual(
                    num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(d2l.Residual(num_channels))
        return blk

    net = nn.Sequential()
    # This model uses a smaller convolution kernel, stride, and padding and
    # removes the maximum pooling layer
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

Previously, we discussed how to use the `initialize` function's `ctx` parameter to initialize model parameters on a CPU or a single GPU. In fact, `ctx` can accept a range of CPUs and GPUs so as to copy initialized model parameters to all CPUs and GPUs in `ctx`.

```{.python .input  n=3}
ctx = d2l.try_all_gpus()
net.initialize(init=init.Normal(sigma=0.01), ctx=ctx)
```

Gluon provides the `split_and_load` function implemented in the previous section. It can divide a mini-batch of data instances and copy them to each CPU or GPU. Then, the model computation for the data input to each CPU or GPU occurs on that same CPU or GPU.

```{.python .input  n=4}
x = nd.random.uniform(shape=(4, 1, 28, 28))
gpu_x = gluon.utils.split_and_load(x, ctx)
net(gpu_x[0]), net(gpu_x[1])
```

Now we can access the initialized model parameter values through `data`. It should be noted that `weight.data()` will return the parameter values on the CPU by default. Since we specified 2 GPUs to initialize the model parameters, we need to specify the GPU to access parameter values. As we can see, the same parameters have the same values on different GPUs.

```{.python .input  n=5}
weight = net[0].params.get('weight')

try:
    weight.data()
except RuntimeError:
    print('not initialized on cpu')
weight.data(ctx[0])[0], weight.data(ctx[1])[0]
```

Remember we define the `evaluate_accuracy_gpu` in :numref:`chapter_lenet` to support evaluating on a single GPU, now we refine this implementation to support multiple devices.

```{.python .input  n=6}
# Save to the d2l package.
def evaluate_accuracy_gpus(net, data_iter):
    # Query the list of devices.
    ctx_list = list(net.collect_params().values())[0].list_ctx()
    acc_sum, n = 0.0, 0
    for features, labels in data_iter:
        Xs, ys = d2l.split_batch(features, labels, ctx_list)
        pys = [net(X) for X in Xs]  # run in parallel
        acc_sum += sum(d2l.accuracy(py, y) for py, y in zip(pys, ys))
        n += features.shape[0]
    return acc_sum / n
```

## Multi-GPU Model Training

When we use multiple GPUs to train the model, the `Trainer` instance will automatically perform data parallelism, such as dividing mini-batches of data instances and copying them to individual GPUs and summing the gradients of each GPU and broadcasting the result to all GPUs. In this way, we can easily implement the training function.

```{.python .input  n=7}
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    ctx_list = [d2l.try_gpu(i) for i in range(num_gpus)]
    net.initialize(init=init.Normal(sigma=0.01),
                   ctx=ctx_list, force_reinit=True)
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd', {'learning_rate': lr})
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 5
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        timer.start()
        for features, labels in train_iter:
            Xs, ys = d2l.split_batch(features, labels, ctx_list)
            with autograd.record():
                ls = [loss(net(X), y) for X, y in zip(Xs, ys)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
        nd.waitall()
        timer.stop()
        animator.add(epoch+1, evaluate_accuracy_gpus(net, test_iter))
    print('test acc: %.2f, %.1f sec/epoch on %s' % (
        animator.Y[0][-1], timer.avg(), ctx_list))
```

First, use a single GPU for training.

```{.python .input  n=8}
train(num_gpus=1, batch_size=256, lr=0.1)
```

Then we try to use 2 GPUs for training. Compared with the LeNet used in the previous section, ResNet-18 computing is more complicated and the communication time is shorter compared to the calculation time, so parallel computing in ResNet-18 better improves performance.

```{.python .input  n=9}
train(num_gpus=2, batch_size=512, lr=0.2)
```

## Summary

* In Gluon, we can conveniently perform multi-GPU computations, such as initializing model parameters and training models on multiple GPUs.

## Exercises

* This section uses ResNet-18. Try different epochs, batch sizes, and learning rates. Use more GPUs for computation if conditions permit.
* Sometimes, different devices provide different computing power. Some can use CPUs and GPUs at the same time, or GPUs of different models. How should we divide mini-batches among different CPUs or GPUs?

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2384)

![](../img/qr_multiple-gpus-gluon.svg)
