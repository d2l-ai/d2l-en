# Networks Using Blocks (VGG)
:label:`chapter_vgg`

While AlexNet proved that deep convolutional neural networks
can achieve good results, it didn't offer a general template
to guide subsequent researchers in designing new networks.
In the following sections, we will introduce several heuristic concepts
commonly used to design deep networks.

Progress in this field mirrors that in chip design
where engineers went from placing transistors
to logical elements to logic blocks.
Similarly, the design of neural network architectures
had grown progressively more abstract,
with researchers moving from thinking in terms of
individual neurons to whole layers,
and now to blocks, repeating patterns of layers.

The idea of using blocks first emerged from the
[Visual Geometry Group](http://www.robots.ox.ac.uk/~vgg/) (VGG)
at Oxford University.
In their eponymously-named VGG network,
It's easy to implement these repeated structures in code
with any modern deep learning framework by using loops and subroutines.


## VGG Blocks

The basic building block of classic convolutional networks
is a sequence of the following layers:
(i) a convolutional layer
(with padding to maintain the resolution),
(ii) a nonlinearity such as a ReLu,
One VGG block consists of a sequence of convolutional layers,
followed by a max pooling layer for spatial downsampling.
In the original VGG paper,
[Simonyan and Ziserman, 2014](https://arxiv.org/abs/1409.1556)
employed convolutions with $3\times3$ kernels
and $2 \times 2$ max pooling with stride of $2$
(halving the resolution after each block).
In the code below, we define a function called `vgg_block`
to implement one VGG block.
The function takes two arguments
corresponding to the number of convolutional layers `num_convs`
and the number of output channels `num_channels`.

```{.python .input  n=1}
import d2l
from mxnet import gluon, nd
from mxnet.gluon import nn

def vgg_block(num_convs, num_channels):
    blk = nn.Sequential()
    for _ in range(num_convs):
        blk.add(nn.Conv2D(num_channels, kernel_size=3,
                          padding=1, activation='relu'))
    blk.add(nn.MaxPool2D(pool_size=2, strides=2))
    return blk
```

## VGG Network

Like AlexNet and LeNet,
the VGG Network can be partitioned into two parts:
the first consisting mostly of convolutional and pooling layers
and a second consisting of fully-connected layers.
The convolutional portion of the net connects several `vgg_block` modules
in succession.
Below, the variable `conv_arch` consists of a list of tuples (one per block),
where each contains two values: the number of convolutional layers
and the number of output channels,
which are precisely the arguments requires to call
the `vgg_block` function.
The fully-connected module is identical to that covered in AlexNet.

![Designing a network from building blocks](../img/vgg.svg)

The original VGG network had 5 convolutional blocks,
among which the first two have one convolutional layer each
and the latter three contain two convolutional layers each.
The first block has 64 output channels
and each subsequent block doubles the number of output channels,
until that number reaches $512$.
Since this network uses $8$ convolutional layers
and $3$ fully-connected layers, it is often called VGG-11.

```{.python .input  n=2}
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
```

The following code implements VGG-11. This is a simple matter of executing a for loop over `conv_arch`.

```{.python .input  n=3}
def vgg(conv_arch):
    net = nn.Sequential()
    # The convolutional layer part
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))
    # The fully connected layer part
    net.add(nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(10))
    return net

net = vgg(conv_arch)
```

Next, we will construct a single-channel data example
with a height and width of 224 to observe the output shape of each layer.

```{.python .input  n=4}
net.initialize()
X = nd.random.uniform(shape=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.name, 'output shape:\t', X.shape)
```

As you can see, we halve height and width at each block,
finally reaching a height and width of 7
before flattening the representations
for processing by the fully-connected layer.

## Model Training

Since VGG-11 is more computationally-heavy than AlexNet
we construct a network with a smaller number of channels.
This is more than sufficient for training on Fashion-MNIST.

```{.python .input  n=5}
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)
```

Apart from using a slightly larger learning rate,
the model training process is similar to that of AlexNet in the last section.

```{.python .input}
lr, num_epochs, batch_size = 0.05, 10, 128,
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch5(net, train_iter, test_iter, num_epochs, lr)
```

## Summary

* VGG-11 constructs a network using reusable convolutional blocks. Different VGG models can be defined by the differences in the number of convolutional layers and output channels in each block.
* The use of blocks leads to very compact representations of the network definition. It allows for efficient design of complex networks.
* In their work Simonyan and Ziserman experimented with various architectures. In particular, they found that several layers of deep and narrow convolutions (i.e. $3 \times 3$) were more effective than fewer layers of wider convolutions.

## Exercises

1. When printing out the dimensions of the layers we only saw 8 results rather than 11. Where did the remaining 3 layer informations go?
1. Compared with AlexNet, VGG is much slower in terms of computation, and it also needs more GPU memory. Try to analyze the reasons for this.
1. Try to change the height and width of the images in Fashion-MNIST from 224 to 96. What influence does this have on the experiments?
1. Refer to Table 1 in the original [VGG Paper](https://arxiv.org/abs/1409.1556) to construct other common models, such as VGG-16 or VGG-19.

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2355)

![](../img/qr_vgg.svg)
