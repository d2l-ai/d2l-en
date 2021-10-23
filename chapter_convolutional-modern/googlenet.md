```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Networks with Parallel Concatenations (GoogLeNet)
:label:`sec_googlenet`

In 2014, *GoogLeNet*
won the ImageNet Challenge :cite:`Szegedy.Liu.Jia.ea.2015`, using a structure
that combined the strengths of NiN :cite:`Lin.Chen.Yan.2013`, repeated blocks :cite:`Simonyan.Zisserman.2014`, 
and a cocktail of convolution kernels. It is arguably also the first network that exhibits a clear distinction between base, stem, and head in a convnet. This design pattern has persisted ever since in the design of deep networks: the *base* is given by the first 2-3 convolutions that operate on the image. They extract low-level features from the underlying images. This is followed by a *stem* of convolutional blocks. Finally, the *head* maps the features obtained so far to the required classification, segmentation, detection, or tracking problem at hand. 

The key contribution in GoogLeNet was the design of the network stem. It solved the problem of selecting 
convolution kernels in an ingenious way. While other works tried to identify which convolution, ranging from $1 \times 1$ to $11 \times 11$ would be best, it simply *combined* multiple convolutions in parallel. 
In what follows we introduce a slightly simplified version of GoogLeNet. The simplifications are due to the fact that  tricks to stabilize training, in particular intermediate loss functions are multiple layers, are no longer needed due to the availability of improved training algorithms. 

## (**Inception Blocks**)

The basic convolutional block in GoogLeNet is called an *Inception block*,
stemming from the meme "we need to go deeper" of the movie *Inception*. 

![Structure of the Inception block.](../img/inception.svg)
:label:`fig_inception`

As depicted in :numref:`fig_inception`,
the inception block consists of four parallel paths.
The first three paths use convolutional layers
with window sizes of $1\times 1$, $3\times 3$, and $5\times 5$
to extract information from different spatial sizes.
The middle two paths also add a $1\times 1$ convolution of the input
to reduce the number of channels, reducing the model's complexity.
The fourth path uses a $3\times 3$ maximum pooling layer,
followed by a $1\times 1$ convolutional layer
to change the number of channels.
The four paths all use appropriate padding to give the input and output the same height and width.
Finally, the outputs along each path are concatenated
along the channel dimension and comprise the block's output.
The commonly-tuned hyperparameters of the Inception block
are the number of output channels per layer.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx, init
from mxnet.gluon import nn
npx.set_np()

class Inception(nn.Block):
    # `c1`--`c4` are the number of output channels for each path
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Path 1
        self.p1_1 = nn.Conv2D(c1, kernel_size=1, activation='relu')
        # Path 2
        self.p2_1 = nn.Conv2D(c2[0], kernel_size=1, activation='relu')
        self.p2_2 = nn.Conv2D(c2[1], kernel_size=3, padding=1,
                              activation='relu')
        # Path 3
        self.p3_1 = nn.Conv2D(c3[0], kernel_size=1, activation='relu')
        self.p3_2 = nn.Conv2D(c3[1], kernel_size=5, padding=2,
                              activation='relu')
        # Path 4
        self.p4_1 = nn.MaxPool2D(pool_size=3, strides=1, padding=1)
        self.p4_2 = nn.Conv2D(c4, kernel_size=1, activation='relu')

    def forward(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        return np.concatenate((p1, p2, p3, p4), axis=1)
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F

class Inception(nn.Module):
    # `c1`--`c4` are the number of output channels for each path
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Path 1
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # Path 2
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # Path 3
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # Path 4
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
from d2l import tensorflow as d2l

class Inception(tf.keras.Model):
    # `c1`--`c4` are the number of output channels for each path
    def __init__(self, c1, c2, c3, c4):
        super().__init__()
        self.p1_1 = tf.keras.layers.Conv2D(c1, 1, activation='relu')
        self.p2_1 = tf.keras.layers.Conv2D(c2[0], 1, activation='relu')
        self.p2_2 = tf.keras.layers.Conv2D(c2[1], 3, padding='same',
                                           activation='relu')
        self.p3_1 = tf.keras.layers.Conv2D(c3[0], 1, activation='relu')
        self.p3_2 = tf.keras.layers.Conv2D(c3[1], 5, padding='same',
                                           activation='relu')
        self.p4_1 = tf.keras.layers.MaxPool2D(3, 1, padding='same')
        self.p4_2 = tf.keras.layers.Conv2D(c4, 1, activation='relu')

    def call(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        return tf.keras.layers.Concatenate()([p1, p2, p3, p4])
```

To gain some intuition for why this network works so well,
consider the combination of the filters.
They explore the image in a variety of filter sizes.
This means that details at different extents
can be recognized efficiently by filters of different sizes.
At the same time, we can allocate different amounts of parameters
for different filters.


## [**GoogLeNet Model**]

As shown in :numref:`fig_inception_full`, GoogLeNet uses a stem of a total of 9 inception blocks, arranged into 3 groups with max-pooling in between, 
and global average pooling in its head to generate its estimates.
Maximum pooling between inception blocks reduces the dimensionality.
At its base, the first module is similar to AlexNet and LeNet.

![The GoogLeNet architecture.](../img/inception-full.svg)
:label:`fig_inception_full`

We can now implement GoogLeNet piece by piece. Let's begin with the base. 
The first module uses a 64-channel $7\times 7$ convolutional layer.

```{.python .input}
%%tab all
class GoogleNet(d2l.Classification):
    def b1(self):
        if tab.selected('mxnet'):
            net = nn.Sequential()
            net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3,
                              activation='relu'),
                    nn.MaxPool2D(pool_size=3, strides=2, padding=1))
            return net
        if tab.selected('pytorch'):
            return nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        if tab.selected('tensorflow'):
            return tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(64, 7, strides=2, padding='same',
                                       activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2, 
                                          padding='same')])
```

The second module uses two convolutional layers:
first, a 64-channel $1\times 1$ convolutional layer,
followed by a $3\times 3$ convolutional layer that triples the number of channels. This corresponds to the second path in the Inception block and concludes the design of the stem. At this point we have 192 channels. 

```{.python .input}
%%tab all
@d2l.add_to_class(GoogleNet)
def b2(self):
    if tab.selected('mxnet'):
        net = nn.Sequential()
        net.add(nn.Conv2D(64, kernel_size=1, activation='relu'),
               nn.Conv2D(192, kernel_size=3, padding=1, activation='relu'),
               nn.MaxPool2D(pool_size=3, strides=2, padding=1))
        return net
    if tab.selected('pytorch'):
        return nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    if tab.selected('tensorflow'):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 1, activation='relu'),
            tf.keras.layers.Conv2D(192, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

The third module connects two complete Inception blocks in series.
The number of output channels of the first Inception block is
$64+128+32+32=256$. This amounts to a 
the ratio of the number of output channel 
among the four paths of $2:4:1:1$. Achieve this, we first reduce the input 
dimensions by $\frac{1}{2}$ and by $\frac{1}{12}$ in the second and third path 
respectively to arrive at $96 = 192/2$ and $16 = 192/12$ channels respectively. 

The number of output channels of the second Inception block
is increased to $128+192+96+64=480$, yielding a ratio of $128:192:96:64 = 4:6:3:2$. As before, 
we need to rduce the number of intermediat dimensions in the second and third channel. A 
scale of $\frac{1}{2}$ and $\frac{1}{8}$ respectively suffices, yielding $128$ and $32$ channels 
respenctively. This captured by the `Inception(64, (96, 128), (16, 32), 32)` and 
`Inception(128, (128, 192), (32, 96), 64)` block constructors. 

```{.python .input}
%%tab all
@d2l.add_to_class(GoogleNet)
def b3(self):
    if tab.selected('mxnet'):
        net = nn.Sequential()
        net.add(Inception(64, (96, 128), (16, 32), 32),
               Inception(128, (128, 192), (32, 96), 64),
               nn.MaxPool2D(pool_size=3, strides=2, padding=1))
        return net
    if tab.selected('pytorch'):
        return nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                             Inception(256, 128, (128, 192), (32, 96), 64),
                             nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    if tab.selected('tensorflow'):
        return tf.keras.models.Sequential([
            Inception(64, (96, 128), (16, 32), 32),
            Inception(128, (128, 192), (32, 96), 64),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

The fourth module is more complicated.
It connects five Inception blocks in series,
and they have $192+208+48+64=512$, $160+224+64+64=512$,
$128+256+64+64=512$, $112+288+64+64=528$,
and $256+320+128+128=832$ output channels, respectively.
The number of channels assigned to these paths is similar
to that in the third module:
the second path with the $3\times 3$ convolutional layer
outputs the largest number of channels,
followed by the first path with only the $1\times 1$ convolutional layer,
the third path with the $5\times 5$ convolutional layer,
and the fourth path with the $3\times 3$ maximum pooling layer.
The second and third paths will first reduce
the number of channels according to the ratio.
These ratios are slightly different in different Inception blocks.

```{.python .input}
%%tab all
@d2l.add_to_class(GoogleNet)
def b4(self):
    if tab.selected('mxnet'):
        net = nn.Sequential()
        net.add(Inception(192, (96, 208), (16, 48), 64),
                Inception(160, (112, 224), (24, 64), 64),
                Inception(128, (128, 256), (24, 64), 64),
                Inception(112, (144, 288), (32, 64), 64),
                Inception(256, (160, 320), (32, 128), 128),
                nn.MaxPool2D(pool_size=3, strides=2, padding=1))
        return net
    if tab.selected('pytorch'):
        return nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                             Inception(512, 160, (112, 224), (24, 64), 64),
                             Inception(512, 128, (128, 256), (24, 64), 64),
                             Inception(512, 112, (144, 288), (32, 64), 64),
                             Inception(528, 256, (160, 320), (32, 128), 128),
                             nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    if tab.selected('tensorflow'):
        return tf.keras.Sequential([
            Inception(192, (96, 208), (16, 48), 64),
            Inception(160, (112, 224), (24, 64), 64),
            Inception(128, (128, 256), (24, 64), 64),
            Inception(112, (144, 288), (32, 64), 64),
            Inception(256, (160, 320), (32, 128), 128),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

The fifth module has two Inception blocks with $256+320+128+128=832$
and $384+384+128+128=1024$ output channels.
The number of channels assigned to each path
is the same as that in the third and fourth modules,
but differs in specific values.
It should be noted that the fifth block is followed by the output layer.
This block uses the global average pooling layer
to change the height and width of each channel to 1, just as in NiN.
Finally, we turn the output into a two-dimensional array
followed by a fully connected layer
whose number of outputs is the number of label classes.

```{.python .input}
%%tab all
@d2l.add_to_class(GoogleNet)
def b5(self):
    if tab.selected('mxnet'):
        net = nn.Sequential()
        net.add(Inception(256, (160, 320), (32, 128), 128),
                Inception(384, (192, 384), (48, 128), 128),
                nn.GlobalAvgPool2D())
        return net
    if tab.selected('pytorch'):
        return nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                             Inception(832, 384, (192, 384), (48, 128), 128),
                             nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())
    if tab.selected('tensorflow'):
        return tf.keras.Sequential([
            Inception(256, (160, 320), (32, 128), 128),
            Inception(384, (192, 384), (48, 128), 128),
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Flatten()])
```

```{.python .input}
%%tab all
@d2l.add_to_class(GoogleNet)
def __init__(self, num_classes=10, lr=0.1):
    super(GoogleNet, self).__init__()
    self.save_hyperparameters()
    if tab.selected('mxnet'):
        self.net = nn.Sequential()
        self.net.add(self.b1(), self.b2(), self.b3(), self.b4(), self.b5(),
                     nn.Dense(num_classes))
        self.net.initialize(init.Xavier())
    if tab.selected('pytorch'):
        self.net = nn.Sequential(self.b1(), self.b2(), self.b3(), self.b4(),
                                 self.b5(), nn.Linear(1024, num_classes))
    if tab.selected('tensorflow'):
        self.net = tf.keras.Sequential([
            self.b1(), self.b2(), self.b3(), self.b4(), self.b5(), 
            tf.keras.layers.Dense(10)])
```

The GoogLeNet model is computationally complex. Note the large number of 
relatively arbitrary parameters in terms of the number of channels chosen. 
This work was done before scientists started using automatic tools to 
optimize network designs. Later on we will see what can be achieved by 
a much more automatic approach network design 
:cite:`howard2019searching,radosavovic2020designing`. 

For now the only modification we will carry out is to 
[**reduce the input height and width from 224 to 96 
to have a reasonable training time on Fashion-MNIST.**]
This simplifies the computation. Let's have a look at the
changes in the shape of the output between the various modules.

```{.python .input}
%%tab mxnet, pytorch
model = GoogleNet().layer_summary((1, 1, 224, 224))
```

```{.python .input}
%%tab tensorflow
model = GoogleNet().layer_summary((1, 224, 224, 1))
```

## [**Training**]

As before, we train our model using the Fashion-MNIST dataset.
 We transform it to $96 \times 96$ pixel resolution
 before invoking the training procedure.

```{.python .input}
%%tab mxnet, pytorch
model = GoogleNet(lr=0.1)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
with d2l.try_gpu():
    model = GoogleNet(lr=0.1)
    trainer.fit(model, data)
```

## Discussion

A key feature of GoogleNet is that is actually *cheaper* to compute than its predecessors 
while simultaneously providing improved accuracy. This marks the beginning of a much more deliberate
network design that trades off the cost of evaluating a network with a reduction in errors. It also marks the beginning of experimentation at a block level with network design parameters, even though it was entirely manual at the time. This is largely due to the fact that DL frameworks in 2015 still lacked much of the design flexibility 
that we now take for granted. Moreover, full network optimization is costly and at the time training on ImageNet still
proved computationally challenging. 

Over the following sections we will encounter a number of design choices (residual connections, batch norm, channel-attention, channel grouping, channel shuffles, pixel shifts, etc.) that allow us to improve networks significantly. For now, you can be proud to have implemented what is arguably the first truly modern convolutional neural network. 

## Exercises

1. GoogLeNet was so successful that it went through a number of iterations. There are several iterations 
   of GoogLeNet that progressively improved speed and accuracy. Try to implement and run some of them. 
   They include the following:
   1. Add a batch normalization layer :cite:`Ioffe.Szegedy.2015`, as described
      later in :numref:`sec_batch_norm`. 
   1. Make adjustments to the Inception block (width, choice and order of convolutions), as described in
      :cite:`Szegedy.Vanhoucke.Ioffe.ea.2016`.
   1. Use label smoothing for model regularization, as described in 
      :cite:`Szegedy.Vanhoucke.Ioffe.ea.2016`.
   1. Make further adjustments to the Inception block by adding residual connection
      :cite:`Szegedy.Ioffe.Vanhoucke.ea.2017`, as described later in
      :numref:`sec_resnet`.
1. What is the minimum image size for GoogLeNet to work?
1. Can you design a variant of GooLeNet that works on FashionMNIST's native resolution of $28 \times 28$ pixels? How 
   would you need to change the base, the stem, and the head of the network, if anything at all?
1. Compare the model parameter sizes of AlexNet, VGG, and NiN with GoogLeNet. How do the latter two network 
   architectures significantly reduce the model parameter size?
1. Compare the amount of computation needed in GooLeNet to AlexNet. How does this affect the design of an accelerator
   chip, e.g. in terms of memory size, amount of computation, the benefit of specialized operations, etc.? 

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/81)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/82)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/316)
:end_tab:

```{.python .input}

```
