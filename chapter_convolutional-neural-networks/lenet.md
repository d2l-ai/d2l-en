# Convolutional Neural Networks (LeNet)

In the ["Implementation of Multilayer Perceptron Starting from Scratch"](../chapter_deep-learning-basics/mlp-scratch.md) section, we construct a multilayer perceptron model with a single hidden layer to classify images in the Fashion-MNIST data set. Both the height and width of each image were 28 pixels. We expand the pixels in the image line by line to get a vector of length 784, and then input it into the fully connected layer. However, this classification method has certain limitations:

1. The adjacent pixels in the same column of an image may be far apart in this vector. The patterns they create may be difficult for the model to recognize.
2. For large input images, using a fully connected layer can easily cause the model to become too large. Suppose the input is a color photo (with three channels) with a height and width of 1000 pixels. Even if the number of outputs of the fully connected layer is still 256, the shape of the weight parameter of the layer is $3,000,000\times 256$, so it takes up about 3 GB of memory. This leads to complex models and excessive storage overhead.

The convolutional layer attempts to solve both problems. On the one hand, the convolutional layer retains the input shape, so that the correlation of image pixels in the directions of both height and width can be recognized effectively. On the other hand, the convolutional layer repeatedly calculates the same kernel and the input of different positions through the sliding window, thereby avoiding excessively large parameter sizes.

A convolutional neural network is a network with convolutional layers. In this section, we will introduce an early convolutional neural network used to recognize handwritten digits in images: LeNet [1]. The name comes from Yann LeCun, the first author of the LeNet paper. LeNet showed that it was possible to use gradient descent to train the convolutional neural network for handwritten digit recognition and achieved outstanding results for its time. This foundational work presented convolutional neural networks to the world for the first time.

## LeNet Model

LeNet is divided into two parts: the convolutional layer block and the fully connected layer block. Below, we will introduce these two modules separately.

The basic units in the convolutional layer block are a convolutional layer and a subsequent maximum pooling layer. The convolutional layer is used to recognize the spatial patterns in the image, such as lines and the parts of objects, and the subsequent maximum pooling layer is used to reduce the sensitivity of the convolutional layer to location. The convolutional layer block is composed of repeated stacks of these two basic units. In the convolutional layer block, each convolutional layer uses a $5\times 5$ window and a sigmoid activation function for the output. The number of output channels for the first convolutional layer is 6, and the number of output channels for the second convolutional layer is increased to 16. This is because the height and width of the input of the second convolutional layer is smaller than that of the first convolutional layer. Therefore, increasing the number of output channels makes the parameter sizes of the two convolutional layers similar. The window shape for the two maximum pooling layers of the convolutional layer block is $2\times 2$ and the stride is 2. Because the pooling window has the same shape as the stride, the areas covered by the pooling window sliding on each input do not overlap.

The output shape of the convolutional layer block is (batch size, channel, height, width). When the output of the convolutional layer block is passed into the fully connected layer block, the fully connected layer block flattens each example in the mini-batch. That is to say, the input shape of the fully connected layer will become two dimensional: the first dimension is the example in the mini-batch, the second dimension is the vector representation after each example is flattened, and the vector length is the product of channel, height, and width.  The fully connected layer block has three fully connected layers. They have 120, 84, and 10 outputs, respectively. Here, 10 is the number of output classes.

Next, we implement the LeNet model through the Sequential class.

```{.python .input}
import gluonbook as gb
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
import time

net = nn.Sequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, activation='sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        # Dense will transform the input of the shape (batch size, channel, height, width) into
        # the input of the shape (batch size, channel *height * width) by default.
        nn.Dense(120, activation='sigmoid'),
        nn.Dense(84, activation='sigmoid'),
        nn.Dense(10))
```

Next, we will construct a single-channel data example with a height and width of 28, and perform a forward computation layer by layer to see the output shape of each layer.

```{.python .input}
X = nd.random.uniform(shape=(1, 1, 28, 28))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

We can see that the height and width of the input in the convolutional layer block is reduced, layer by layer. The convolutional layer uses a kernel with a height and width of 5 to reduce the height and width by 4, while the pooling layer halves the height and width, but the number of channels increases from 1 to 16. The fully connected layer reduces the number of outputs layer by layer, until the number of image classes becomes 10.


## Data Acquisition and Training

Now, we will experiment with the LeNet model. In this experiment, we still use Fashion-MNIST as the training data set.

```{.python .input}
batch_size = 256
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size=batch_size)
```

Since convolutional neural network computing is more complex than multilayer perceptrons, we recommend using GPUs to speed up computing. We try to create NDArray on `gpu(0)`, and use `gpu(0)` if this is successful. Otherwise, we will stick with CPU.

```{.python .input}
def try_gpu4():  # This function has been saved in the gluonbook package for future use.
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx

ctx = try_gpu4()
ctx
```

Accordingly, we slightly modify the `evaluate_accuracy` function described in the ["Implementation of Softmax Regression Starting from Scratch"](../chapter_deep-learning-basics/softmax-regression-scratch.md) section.  Because, at first, the data is stored on the CPU's memory, when the `ctx` variable is GPU, we copy the data to the GPU through the `as_in_context` function described in the ["GPU Computing"](../chapter_deep-learning-computation/use-gpu.md) section. For example, we can use `gpu(0)`. 

```{.python .input}
# This function has been saved in the gluonbook package for future use. The function will be gradually improved. Its complete implementation will be
# discussed in the "Image Augmentation" section.
def evaluate_accuracy(data_iter, net, ctx):
    acc = nd.array([0], ctx=ctx)
    for X, y in data_iter:
        # If ctx is the GPU, copy the data to the GPU.
        X, y = X.as_in_context(ctx), y.as_in_context(ctx)
        acc += gb.accuracy(net(X), y)
    return acc.asscalar() / len(data_iter)
```

We also slightly modify the `train_ch3` function defined in the ["Implementation of Softmax Regression Starting from Scratch"](../chapter_deep-learning-basics/softmax-regression-scratch.md) section, ensuring that the data and model used in computing are on the CPU or GPU memory.

```{.python .input}
# This function has been saved in the gluonbook package for future use.
def train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx,
              num_epochs):
    print('training on', ctx)
    loss = gloss.SoftmaxCrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, start = 0, 0, time.time()
        for X, y in train_iter:
            X, y = X.as_in_context(ctx), y.as_in_context(ctx)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(batch_size)
            train_l_sum += l.mean().asscalar()
            train_acc_sum += gb.accuracy(y_hat, y)
        test_acc = evaluate_accuracy(test_iter, net, ctx)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
              'time %.1f sec' % (epoch + 1, train_l_sum / len(train_iter),
                                 train_acc_sum / len(train_iter),
                                 test_acc, time.time() - start))
```

We initialize the model parameters onto the device variable `ctx` again, and initialize them randomly using Xavier. The loss function and the training algorithm still use the cross-entropy loss function and mini-batch stochastic gradient descent.

```{.python .input}
lr, num_epochs = 0.9, 5
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)
```

## Summary

* A convolutional neural network is a network with convolutional layers.
* LeNet uses alternating convolutional layers and maximum pooling layers, followed by fully connected layers for image classification.

## exercise

* Try to construct a more complex network based on LeNet to improve its accuracy. For example, adjust the convolution window size, the number of output channels, the activation function, and the number of fully connected layer outputs. In terms of optimization, you can try using different learning rates, initialization methods, and increasing the epochs.


## Scan the QR Code to Access [Discussions](https://discuss.gluon.ai/t/topic/737)

![](../img/qr_lenet.svg)

## References

[1] LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.
