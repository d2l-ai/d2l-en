# Convolutional Neural Networks (LeNet)

In our first encounter with image data we applied a [Multilayer Perceptron](../chapter_deep-learning-basics/mlp-scratch.md) to pictures of clothing in the Fashion-MNIST data set. Both the height and width of each image were 28 pixels. We expanded the pixels in the image line by line to get a vector of length 784, and then used them as inputs to the fully connected layer. However, this classification method has certain limitations:

1. The adjacent pixels in the same column of an image may be far apart in this vector. The patterns they create may be difficult for the model to recognize. In fact, the vectorial representation ignores position entirely - we could have permuted all $28 \times 28$ pixels at random and obtained the same results.
2. For large input images, using a fully connected layer can easily cause the model to become too large, as we discussed previously.

As discussed in the previous sections, the convolutional layer attempts to solve both problems. On the one hand, the convolutional layer retains the input shape, so that the correlation of image pixels in the directions of both height and width can be recognized effectively. On the other hand, the convolutional layer repeatedly calculates the same kernel and the input of different positions through the sliding window, thereby avoiding excessively large parameter sizes.

A convolutional neural network is a network with convolutional layers. In this section, we will introduce an early convolutional neural network used to recognize handwritten digits in images - [LeNet5](http://yann.lecun.com/exdb/lenet/). Convolutional networks were invented by Yann LeCun and coworkers at AT&T Bell Labs in the early 90s. LeNet showed that it was possible to use gradient descent to train the convolutional neural network for handwritten digit recognition. It achieved outstanding results at the time (only matched by Support Vector Machines at the time).

## LeNet

LeNet is divided into two parts: a block of convolutional layers and one of fully connected ones. Below, we will introduce these two modules separately. Before going into details, let's briefly review the model in pictures. To illustrate the issue of channels and the specific layers we will use a rather description (later we will see how to convey the same information more concisely).

![Data flow in LeNet 5. The input is a handwritten digit, the output a probabilitiy over 10 possible outcomes.](../img/lenet.svg)

The basic units in the convolutional block are a convolutional layer and a subsequent average pooling layer (note that max-pooling works better, but it had not been invented in the 90s yet). The convolutional layer is used to recognize the spatial patterns in the image, such as lines and the parts of objects, and the subsequent average pooling layer is used to reduce the dimensionality. The convolutional layer block is composed of repeated stacks of these two basic units. In the convolutional layer block, each convolutional layer uses a $5\times 5$ window and a sigmoid activation function for the output (note that ReLu works better, but it had not been invented in the 90s yet). The number of output channels for the first convolutional layer is 6, and the number of output channels for the second convolutional layer is increased to 16. This is because the height and width of the input of the second convolutional layer is smaller than that of the first convolutional layer. Therefore, increasing the number of output channels makes the parameter sizes of the two convolutional layers similar. The window shape for the two average pooling layers of the convolutional layer block is $2\times 2$ and the stride is 2. Because the pooling window has the same shape as the stride, the areas covered by the pooling window sliding on each input do not overlap. In other words, the pooling layer performs downsampling.

The output shape of the convolutional layer block is (batch size, channel, height, width). When the output of the convolutional layer block is passed into the fully connected layer block, the fully connected layer block flattens each example in the mini-batch. That is to say, the input shape of the fully connected layer will become two dimensional: the first dimension is the example in the mini-batch, the second dimension is the vector representation after each example is flattened, and the vector length is the product of channel, height, and width.  The fully connected layer block has three fully connected layers. They have 120, 84, and 10 outputs, respectively. Here, 10 is the number of output classes.

Next, we implement the LeNet model through the Sequential class.

```{.python .input}
import gluonbook as gb
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
import time

net = nn.Sequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, padding=2, activation='sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        # Dense will transform the input of the shape (batch size, channel, height, width) into
        # the input of the shape (batch size, channel *height * width) automatically by default.
        nn.Dense(120, activation='sigmoid'),
        nn.Dense(84, activation='sigmoid'),
        nn.Dense(10))
```

We took the liberty of replacing the Gaussian activation in the last layer by a regular dense network since this is rather much more convenient to train. Other than that the network matches the historical definition of LeNet5. Next, we feed a single-channel example of size $28 \times 28$ into the network and perform a forward computation layer by layer to see the output shape of each layer.

```{.python .input}
X = nd.random.uniform(shape=(1, 1, 28, 28))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

We can see that the height and width of the input in the convolutional layer block is reduced, layer by layer. The convolutional layer uses a kernel with a height and width of 5 to reduce the height and width by 4, while the pooling layer halves the height and width, but the number of channels increases from 1 to 16. The fully connected layer reduces the number of outputs layer by layer, until the number of image classes becomes 10.

![Compressed notation for LeNet5](../img/lenet-vert.svg)


## Data Acquisition and Training

Now, we will experiment with the LeNet model. We still use Fashion-MNIST as the training data set since the problem is rather more difficult than OCR (even in the 1990s the error rates were in the 1% range).

```{.python .input}
batch_size = 256
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size=batch_size)
```

Since convolutional networks are significantly more expensive to compute than multilayer perceptrons we recommend using GPUs to speed up training. Time to introduce a convenience function that allows us to detect whether we have a GPU: it works by trying to allocate an NDArray on `gpu(0)`, and use `gpu(0)` if this is successful. Otherwise, we catch the resulting exception and we stick with the CPU.

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

Accordingly, we slightly modify the `evaluate_accuracy` function described when [implementing the SoftMax from scratch](../chapter_deep-learning-basics/softmax-regression-scratch.md).  Since the data arrives in the CPU when loading we need to copy it to the GPU before any computation can occur. This is accomplished via the `as_in_context` function described in the [GPU Computing](../chapter_deep-learning-computation/use-gpu.md) section. Note that we accumulate the errors on the same device as where the data eventually lives (in `acc`). This avoids intermediate copy operations that would destroy performance.

```{.python .input}
# This function has been saved in the gluonbook package for future use. The function will be gradually improved.
# Its complete implementation will be discussed in the "Image Augmentation" section.
def evaluate_accuracy(data_iter, net, ctx):
    acc = nd.array([0], ctx=ctx)
    for X, y in data_iter:
        # If ctx is the GPU, copy the data to the GPU.
        X, y = X.as_in_context(ctx), y.as_in_context(ctx)
        acc += gb.accuracy(net(X), y)
    return acc.asscalar() / len(data_iter)
```

Just like the data loader we need to update the training function to deal with GPUs. Unlike [`train_ch3`](../chapter_deep-learning-basics/softmax-regression-scratch.md) we now move data prior to computation.

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

We initialize the model parameters on the device indicated by `ctx`, this time using Xavier. The loss function and the training algorithm still use the cross-entropy loss function and mini-batch stochastic gradient descent.

```{.python .input}
lr, num_epochs = 0.9, 5
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)
```

## Summary

* A convolutional neural network (in short, ConvNet) is a network using convolutional layers.
* In a ConvNet we alternate between convolutions, nonlinearities and often also pooling operations.
* Ultimately the resolution is reduced prior to emitting an output via one (or more) dense layers.
* LeNet was the first successful deployment of such a network.

## Problems

1. Replace the average pooling with max pooling. What happens?
1. Try to construct a more complex network based on LeNet to improve its accuracy.
    * Adjust the convolution window size.
    * Adjust the number of output channels.
    * Adjust the activation function (ReLu?).
    * Adjust the number of convolution layers.
    * Adjust the number of fully connected layers.
    * Adjust the learning rates and other training details (initialization, epochs, etc.)
1. Try out the improved network on the original MNIST dataset.
1. Display the activations of the first and second layer of LeNet for different inputs (e.g. sweaters, coats).


## References

[1] LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.

## Discuss on our Forum

<div id="discuss" topic_id="2353"></div>
