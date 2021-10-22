```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Networks Using Blocks (VGG)
:label:`sec_vgg`

While AlexNet offered empirical evidence that deep CNNs
can achieve good results, it did not provide a general template
to guide subsequent researchers in designing new networks.
In the following sections, we will introduce several heuristic concepts
commonly used to design deep networks.

Progress in this field mirrors that of VLSI (Very Large Scale Integration) 
in chip design
where engineers moved from placing transistors
to logical elements to logic blocks :cite:`mead1980introduction`.
Similarly, the design of neural network architectures
has grown progressively more abstract,
with researchers moving from thinking in terms of
individual neurons to whole layers,
and now to blocks, repeating patterns of layers. 

The idea of using blocks first emerged from the
[Visual Geometry Group](http://www.robots.ox.ac.uk/~vgg/) (VGG)
at Oxford University,
in their eponymously-named *VGG* network :cite:`Simonyan.Zisserman.2014`.
It is easy to implement these repeated structures in code
with any modern deep learning framework by using loops and subroutines.

## (**VGG Blocks**)
:label:`subsec_vgg-blocks`

The basic building block of CNNs
is a sequence of the following:
(i) a convolutional layer
with padding to maintain the resolution,
(ii) a nonlinearity such as a ReLU,
(iii) a pooling layer such
as max-pooling to reduce the resolution. One of the problems with 
this approach is that the spatial resolution decreases quite rapidly. In particular, 
this imposes a hard limit of $\log_2 d$ convolutional layers on the network before all 
dimensions are used up. For instance, in the case of ImageNet, it would be impossible to have 
more than 8 convolutional layers in this way. 

The key idea by Simonyan and Zisserman was to use *multiple* convolutions in between downsampling
via max-pooling in the form of a block. They were primarily interested in whether deep or 
wide networks perform better. For instance, the successive application of two $3 \times 3$ convolutions
touches the same pixels as a single $5 \times 5$ convolution does. At the same time, the latter uses approximately 
as many parameters ($25 \cdot c^2$) as three $3 \times 3$ comvolutions do ($3 \cdot 9 \cdot c^2$). 
In a rather detailed analysis they showed that deep and narrow networks significantly outperform their shallow counterparts. This set deep learning on a quest for ever deeper networks with over 100 layers for typical applications. 

Back to VGG: a VGG block consists of a *sequence* of convolutions with $3\times3$ kernels with padding of 1 
(keeping height and width) followed by a $2 \times 2$ max-pooling layer with stride of 2
(halving the resolution after each block).
In the code below, we define a function called `vgg_block`
to implement one VGG block.

:begin_tab:`mxnet`
The function below takes two arguments,
corresponding to the number of convolutional layers `num_convs`
and the number of output channels `num_channels`.
:end_tab:

:begin_tab:`pytorch`
The function below takes three arguments corresponding to the number
of convolutional layers `num_convs`, the number of input channels `in_channels`
and the number of output channels `out_channels`.
:end_tab:

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx, init
from mxnet.gluon import nn
npx.set_np()

def vgg_block(num_convs, num_channels):
    blk = nn.Sequential()
    for _ in range(num_convs):
        blk.add(nn.Conv2D(num_channels, kernel_size=3,
                          padding=1, activation='relu'))
    blk.add(nn.MaxPool2D(pool_size=2, strides=2))
    return blk
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
from d2l import tensorflow as d2l

def vgg_block(num_convs, num_channels):
    blk = tf.keras.models.Sequential()
    for _ in range(num_convs):
        blk.add(
            tf.keras.layers.Conv2D(num_channels, kernel_size=3,
                                   padding='same', activation='relu'))
    blk.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    return blk
```

## [**VGG Network**]
:label:`subsec_vgg-network`

Like AlexNet and LeNet, 
the VGG Network can be partitioned into two parts:
the first consisting mostly of convolutional and pooling layers
and the second consisting of fully connected layers that are identical to those in AlexNet. 
The key difference is 
that the convolutional layers are grouped in nonlinear transformations that 
leave the dimensonality unchanged, followed by a resolution-reduction step, as 
depicted in :numref:`fig_vgg`. 

![From AlexNet to VGG that is designed from building blocks.](../img/vgg.svg)
:width:`400px`
:label:`fig_vgg`

The convolutional part of the network connects several VGG blocks from :numref:`fig_vgg` (also defined in the `vgg_block` function)
in succession. This grouping of convolutions is a pattern that has 
remained almost unchanged over the past decade, although the specific choice of 
operations has undergone considerable modifications. 
The variable `conv_arch` consists of a list of tuples (one per block),
where each contains two values: the number of convolutional layers
and the number of output channels,
which are precisely the arguments required to call
the `vgg_block` function. As such, VGG defines a *family* of networks rather than just 
a specific manifestation. To build a specific network we simply iterate over `conv_arch` to compose the blocks.

```{.python .input}
%%tab all
class VGG(d2l.Classification):
    def __init__(self, arch, lr=0.1):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            for (num_convs, num_channels) in arch:
                self.net.add(vgg_block(num_convs, num_channels))
            self.net.add(nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                         nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                         nn.Dense(10))
            self.net.initialize(init.Xavier())
        if tab.selected('pytorch'):
            conv_blks = []
            in_channels = 1
            for (num_convs, out_channels) in arch:
                conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
                in_channels = out_channels
            self.net = nn.Sequential(
                *conv_blks, nn.Flatten(),
                nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(4096, 10))
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential()
            for (num_convs, num_channels) in arch:
                self.net.add(vgg_block(num_convs, num_channels))
            self.net.add(
                tf.keras.models.Sequential([
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(4096, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(4096, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(10)]))
```

The original VGG network had 5 convolutional blocks,
among which the first two have one convolutional layer each
and the latter three contain two convolutional layers each.
The first block has 64 output channels
and each subsequent block doubles the number of output channels,
until that number reaches 512.
Since this network uses 8 convolutional layers
and 3 fully connected layers, it is often called VGG-11.

```{.python .input}
%%tab pytorch, mxnet
VGG(arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))).layer_summary((1, 1, 224, 224))
```

```{.python .input}
%%tab tensorflow
VGG(arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))).layer_summary((1, 224, 224, 1))
```

As you can see, we halve height and width at each block,
finally reaching a height and width of 7
before flattening the representations
for processing by the fully connected part of the network.

## Training

[**Since VGG-11 is more computationally-heavy than AlexNet
we construct a network with a smaller number of channels.**]
This is more than sufficient for training on Fashion-MNIST.

Apart from using a slightly larger learning rate,
the [**model training**] process is similar to that of AlexNet in :numref:`sec_alexnet`.

```{.python .input}
%%tab mxnet, pytorch
model = VGG(arch=((1, 16), (1, 32), (2, 64), (2, 128), (2, 128)), lr=0.05)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
with d2l.try_gpu():
    model = VGG(arch=((1, 16), (1, 32), (2, 64), (2, 128), (2, 128)), lr=0.05)
    trainer.fit(model, data)
```

## Summary

One might argue that VGG is the first truly modern convolutional neural network. While AlexNet introduced many of the components of what make deep learning effective at scale, it is VGG that arguably introduced key properties such as blocks of multiple convolutions and a preference for deep and narrow networks. It is also the first network that is actually an entire family of similarly parametrized models, giving the practitioner ample trade-off between complexity and speed. This is also the place where modern DL frameworks shine. It is no longer necessary to generate XML config files to specify a network but rather, to assmple said networks through simple Python code. 

Very recently ParNet :cite:`goyal2021non` demonstrated that it is possible to achieve competitive performance using a much more shallow architecture through a large number of parallel computations. This is an exciting development and there's hope that it will influence architecture designs in the future. For the remainder of the chapter, though, we will follow the path of scientific progress over the past decade. 

## Exercises


1. Compared with AlexNet, VGG is much slower in terms of computation, and it also needs more GPU memory. 
    1. Compare the number of parameters needed for AlexNet and VGG.
    1. Compare the number of floating point operations used in the convolutional layers and in the dense layers. 
    1. How could you reduce the computational cost created by the dense layers?
1. When displaying the dimensions associated with the various layers of the network, we only see the information 
   associated with 8 blocks (plus some auxiliary transforms), even though the network has 11 layers. Where did 
   the remaining 3 layers go?
1. Upsampling the resolution in Fashion MNIST by a factor of $8 \times 8$ from 28 to 224 dimensions is highly 
   wasteful. Try modifying the network architecture and resolution conversion, e.g., to 56 or to 84 dimensions 
   for its input instead. Can you do so without reducing the accuracy of the network?
1. Use Table 1 in the VGG paper :cite:`Simonyan.Zisserman.2014` to construct other common models, 
   such as VGG-16 or VGG-19.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/77)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/78)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/277)
:end_tab:
