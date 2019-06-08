# Convolutional Neural Networks (LeNet)
:label:`chapter_lenet`

We are now ready to put all of the tools together
to deploy your first fully-functional convolutional neural network.
In our first encounter with image data we applied a multilayer perceptron (:numref:`chapter_mlp_scratch`)
to pictures of clothing in the Fashion-MNIST data set.
Each image in Fashion-MNIST consisted of
a two-dimensional $28 \times 28$ matrix.
To make this data amenable to multilayer perceptrons
which anticapte receiving inputs as one-dimensional fixed-length vectors,
we first flattened each image, yielding vectors of length 784,
before processing them with a series of fully-connected layers.

Now that we have introduced convolutional layers,
we can keep the image in its original spatially-organized grid,
processing it with a series of successive convolutional layers.
Moreover, because we are using convolutional layers,
we can enjoy a considerable savings in the number of parameters required.

In this section, we will introduce one of the first
published convolutional neural networks
whose benefit was first demonstrated by Yann Lecun,
then a researcher at AT&T Bell Labs,
for the purpose of recognizing handwritten digits in images—[LeNet5](http://yann.lecun.com/exdb/lenet/).
In the 90s, their experiments with LeNet gave the first compelling evidence
that it was possible to train convolutional neural networks
by backpropagation.
Their model achieved outstanding results at the time
(only matched by Support Vector Machines at the time)
and was adopted to recognize digits for processing deposits in ATM machines.
Some ATMs still runn the code
that Yann and his colleague Leon Bottou wrote in the 1990s!

## LeNet

In a rough sense, we can think LeNet as consisting of two parts:
(i) a block of convolutional layers; and
(ii) a block of fully-connected layers.
Before getting into the weeds, let's briefly review the model in :numref:`img_lenet`.

![Data flow in LeNet 5. The input is a handwritten digit, the output a probabilitiy over 10 possible outcomes.](../img/lenet.svg)
:label:`img_lenet`

The basic units in the convolutional block are a convolutional layer
and a subsequent average pooling layer
(note that max-pooling works better,
but it had not been invented in the 90s yet).
The convolutional layer is used to recognize
the spatial patterns in the image,
such as lines and the parts of objects,
and the subsequent average pooling layer
is used to reduce the dimensionality.
The convolutional layer block is composed of
repeated stacks of these two basic units.
Each convolutional layer uses a $5\times 5$ kernel
and processes each output with a sigmoid activation function
(again, note that ReLUs are now known to work more reliably,
but had not been invented yet).
The first convolutional layer has 6 output channels,
and second convolutional layer increases channel depth further to 16.

However, coinciding with this increase in the number of channels,
the height and width are shrunk considerably.
Therefore, increasing the number of output channels
makes the parameter sizes of the two convolutional layers similar.
The two average pooling layers are of size $2\times 2$ and take stride 2
(note that this means they are non-overlapping).
In other words, the pooling layer downsamples the representation
to be precisely *one quarter* the pre-pooling size.

The convolutional block emits an output with size given by
(batch size, channel, height, width).
Before we can pass the convolutional block's output
to the fully-connected block, we must flatten
each example in the mini-batch.
In other words, we take this 4D input and tansform it into the 2D
input expected by fully-connected layers:
as a reminder, the first dimension indexes the examples in the mini-batch
and the second gives the flat vector representation of each example.
LeNet's fully-connected layer block has three fully-connected layers,
with 120, 84, and 10 outputs, respectively.
Because we are still performing classification,
the 10 dimensional output layer corresponds
to the number of possible output classes.

While getting to the point
where you truly understand
what's going on inside LeNet
may have taken a bit of work,
you can see below that implementing it
in a modern deep learning library
is remarkably simple.
Again, we'll rely on the Sequential class.

```{.python .input}
import d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import nn
import time

net = nn.Sequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, padding=2, activation='sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        # Dense will transform the input of the shape (batch size, channel,
        # height, width) into the input of the shape (batch size,
        # channel * height * width) automatically by default
        nn.Dense(120, activation='sigmoid'),
        nn.Dense(84, activation='sigmoid'),
        nn.Dense(10))
```

As compared to the original network,
we took the liberty of replacing
the Gaussian activation in the last layer
by a regular dense layer, which tends to be
significantly more convenient to train.
Other than that, this network matches
the historical definition of LeNet5.
Next, we feed a single-channel example
of size $28 \times 28$ into the network
and perform a forward computation layer by layer
printing the output shape at each layer
to make sure we understand what's happening here.

```{.python .input}
X = nd.random.uniform(shape=(1, 1, 28, 28))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

Note that the height and width of the representation
at each layer throughout the convolutional block is reduced
(compared to the previous layer).
The convolutional layer uses a kernel
with a height and width of 5,
which with only $2$ pixels of padding in the first convolutional layer
and none in the second convolutional layer
leads to reductions in both height and width by 2 and 4 pixels, respectively.
Moreover each pooling layer halves the height and width.
However, as we go up the stack of layers,
the number of channels increases layer-over-layer
from 1 in the input to 6 after the first convolutional layer
and 16 after the second layer.
Then, the fully-connected layer reduces dimensionality layer by layer,
until emitting an output that matches the number of image classes.

![Compressed notation for LeNet5](../img/lenet-vert.svg)


## Data Acquisition and Training

Now that we've implemented the model,
we might as well run some experiments
to see what we can accomplish with the LeNet model.
While it might serve nostalgia
to train LeNet on the original MNIST dataset,
that dataset has become too easy,
with MLPs getting over 98% accuracy,
so it would be hard to see the benefits of convolutional networks.
Thus we will stick with Fashion-MNIST as our dataset
because while it has the same shape ($28\times28$ images),
this dataset is notably more challenging.

```{.python .input}
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
```

While convolutional networks may have few parameters,
they can still be significantly more expensive
to compute than a similarly deep multilayer perceptron
so if you have access to a GPU, this might be a good time
to put it into action to speed up training.

For evaluation, we need to make a slight modification
to the `evaluate_accuracy` function that we described
in :numref:`chapter_softmax_scratch`.
Since the full dataset lives on the CPU,
we need to copy it to the GPU before we can compute our models.
This is accomplished via the `as_in_context` function
described in :numref:`chapter_use_gpu`. This implementation will overwrite the one introduced from :numref:`chapter_softmax_scratch` in the `d2l` package.

```{.python .input}
# Save to the d2l package
def evaluate_accuracy(net, data_iter):
    # Query on which device the parameter is.
    ctx = list(net.collect_params().values())[0].list_ctx()[0]
    acc_sum, n = nd.array([0], ctx=ctx), 0
    for X, y in data_iter:
        X, y = X.as_in_context(ctx), y.as_in_context(ctx).astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y).sum()
        n += y.size
    return acc_sum.asscalar() / n
```

We also need to update our training function to deal with GPUs.
Unlike the `train_epoch_ch3` defined in :numref:`chapter_softmax_scratch`, we now need to move each batch of data to our designated context (hopefully, the GPU)
prior to making the forward and backward passes.

```{.python .input}
# Save to the d2l package.
def train_epoch_ch5(net, train_iter, loss, trainer, ctx):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:
        # Here is the only difference compared to train_epoch_ch3
        X, y = X.as_in_context(ctx), y.as_in_context(ctx)
        with autograd.record():
            y_hat = net(X)
            l = loss(y_hat, y)
        l.backward()
        trainer.step(X.shape[0])
        train_l_sum += l.sum().asscalar()
        train_acc_sum += d2l.accuracy(y_hat, y)
        n += X.shape[0]
    return train_l_sum / n, train_acc_sum / n
```

The training function `train_ch5` is also very similar to `train_ch3` defined in :numref:`chapter_softmax_scratch`. Since we will deal with networks with tens of layers now, the function will only support Gluon models. We initialize the model parameters on the device indicated by `ctx`,
this time using the Xavier initializer.
The loss function and the training algorithm
still use the cross-entropy loss function
and mini-batch stochastic gradient descent.

```{.python .input}
# Save to the d2l package.
def train_ch5(net, train_iter, test_iter, num_epochs, lr, 
              ctx=d2l.try_gpu()):
    net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(),
                            'sgd', {'learning_rate': lr})
    trains, test_accs = [], []
    d2l.use_svg_display()
    fig, ax = d2l.plt.subplots(figsize=(4, 3))
    start = time.time()
    for epoch in range(num_epochs):
        trains.append(train_epoch_ch5(
            net, train_iter, loss, trainer, ctx))
        test_accs.append(evaluate_accuracy(net, test_iter))
        legend = ['train loss', 'train acc', 'test acc']
        res = list(map(list, zip(*trains)))+[test_accs,]
        d2l.plot(list(range(1, epoch+2)), res, 'epoch', legend=legend, 
             xlim=[0,num_epochs+1], ylim=[0, 1], axes=ax)
        d2l.show(fig)
    print('Done in %d sec on %s, loss %.3f, train acc %.3f, test acc %.3f'%(
        time.time()-start, ctx, *trains[-1], test_accs[-1]))
```

Now let's train the model.

```{.python .input}
lr, num_epochs = 0.9, 10
train_ch5(net, train_iter, test_iter, num_epochs, lr)
```

## Summary

* A convolutional neural network (in short, ConvNet) is a network using convolutional layers.
* In a ConvNet we alternate between convolutions, nonlinearities and often also pooling operations.
* Ultimately the resolution is reduced prior to emitting an output via one (or more) dense layers.
* LeNet was the first successful deployment of such a network.

## Exercises

1. Replace the average pooling with max pooling. What happens?
1. Try to construct a more complex network based on LeNet to improve its accuracy.
    * Adjust the convolution window size.
    * Adjust the number of output channels.
    * Adjust the activation function (ReLU?).
    * Adjust the number of convolution layers.
    * Adjust the number of fully connected layers.
    * Adjust the learning rates and other training details (initialization, epochs, etc.)
1. Try out the improved network on the original MNIST dataset.
1. Display the activations of the first and second layer of LeNet for different inputs (e.g. sweaters, coats).


## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2353)

![](../img/qr_lenet.svg)
