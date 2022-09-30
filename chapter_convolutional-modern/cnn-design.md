```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Designing Convolution Network Architectures
:label:`sec_cnn-design`

The past sections took us on a tour of modern network design for computer vision. Common to all the work we covered was that it heavily relied on the intuition of scientists. Many of the architectures are heavily informed by human creativity and to a much lesser extent by systematic exploration of the design space that deep nets offer. Nonetheless, this *network engineering* approach has been tremendously successful. 

Since AlexNet (:numref:`sec_alexnet`)
beat conventional computer vision models on ImageNet,
it became popular to construct very deep networks
by stacking blocks of convolutions, all designed by the same pattern. 
In particular, $3 \times 3$ convolutions were 
popularized by VGG networks (:numref:`sec_vgg`).
NiN (:numref:`sec_nin`) showed that even $1 \times 1$ convolutions could 
be beneficial by adding local nonlinearities. 
Moreover, NiN solved the problem of aggregating information at the head of a network 
by aggregation across all locations. 
GoogLeNet (:numref:`sec_googlenet`) added multiple branches of different convolution width, 
combining the advantages of VGG and NiN in its Inception block. 
ResNets (:numref:`sec_resnet`) changed the inductive bias towards the identity mapping (from $f(x) = 0$). This allowed for very deep networks. Almost a decade later, the ResNet design is still popular, a testament to its design. Lastly, ResNeXt (:numref:`sec_resnext`) added grouped convolutions, offering a better trade-off between parameters and computation. A precursor to Transformers for vision, the Squeeze-and-Excitation Networks (SENets) allow for efficient information transfer between locations. 
:cite:`Hu.Shen.Sun.2018`. They accomplish this by computing a per-channel global attention function. 

So far we omitted networks obtained via *neural architecture search* (NAS) :cite:`zoph2016neural,liu2018darts`. We chose to do so since their cost is usually enormous, relying on brute force search, genetic algorithms, reinforcement learning, or some other form of hyperparameter optimization. Given a fixed search space,
NAS uses a search strategy to automatically select
an architecture based on the returned performance estimation.
The outcome of NAS
is a single network instance. EfficientNets are a notable outcome of this search :cite:`tan2019efficientnet`.

In the following we discuss an idea that is quite different to the quest for the *single best network*. It is computationally relatively inexpensive, it leads to scientific insights on the way, and it is quite effective in terms of the quality of outcomes. Let's review the strategy by :cite:`Radosavovic.Kosaraju.Girshick.ea.2020` to *design network design spaces*. The strategy combines the strength of manual design and NAS. It accomplishes this by operating on *distributions of networks* and optimizing the distributions in a way to obtain good performance for entire families of networks. The outcome of it are *RegNets*, specifically RegNetX and RegNetY, plus a range of guiding principles for the design of performant Convnets. 

## The AnyNet Design Space

The description below closely follows the reasoning in :cite:`Radosavovic.Kosaraju.Girshick.ea.2020` with some abbreviations to make it fit in the scope of the book. We recommend the interested reader to peruse the original publication for further detail. We need a template for the family of networks to explore. One of the commonalities of the designs in this chapter is that the networks consist of a *stem*, a *body* and a *head*. The stem performs initial image processing, often through convolutions with a larger window size. The body consists of multiple blocks, carrying out the bulk of the transformations needed to go from raw images to object representations. Lastly, the *head* converts this into the desired outputs, such as via a logistic regressor for multiclass classification. 
The body, in turn, consists of multiple stages, operating on the image at decreasing resolutions. In fact, both the stem and each subsequent stage quarter the spatial resolution. Lastly, each stage consists of one or more blocks. This pattern is common to all networks, from VGG to ResNeXt. Indeed, for the design of generic AnyNet networks, :citet:`Radosavovic.Kosaraju.Girshick.ea.2020` use the ResNeXt block of :numref:`fig_resnext_block`. 

![The AnyNet design. The numbers $(c,r)$ at the top of each module indicate the number of channels $c$ and the resolution $r \times r$ of the images at that point. From left to right:  generic network structure, composed of stem, body and head; 
body composed of multiple stages; detailed structure of a stage; To the right we see two alternative structures for blocks, one without downsampling and one that halves the resolution in each dimension.](../img/anynet-full.svg)
:label:`fig_anynet_full`

Let's review the structure outlined in :numref:`fig_anynet_full` in detail. As mentioned, an AnyNet consists of a stem, body and head. The stem takes as its input RGB images (3 channels), using a convolution with a stride of $2$, followed by a batch norm, to halve the resolution from $r \times r$ to $r/2 \times r/2$. Moreover, it generates $c_0$ channels that serve as input to the body. 

Since the network is designed to work well with ImageNet images of $224 \times 224 \times 3$ resolution, the body serves to reduce this to $7 \times 7 \times c_4$ through 4 stages, each with an eventual stride of $2$. Lastly, the head employs an entirely standard design via global average pooling, similar to NiN :numref:`sec_nin`, followed by a fully connected layer to emit an $n$-dimensional vector for $n$-class classification. 

Most of the relevant design decisions are inherent to the body of the network. It proceeds in stages, where each stage is composed of the same type of ResNeXt blocks as we discussed in :numref:`sec_resnet`. The design there is again entirely generic: we begin with a block that halves the resolution by using a stride of $2$ (the rightmost network design). To match this, the residual branch of the ResNeXt block needs to pass through a $1 \times 1$ convolution. This block is followed by a variable number of additional ResNeXt blocks that leave both resolution and number of channels unchanged. Note that a common design practice is to add a slight bottleneck in the design of convolutional blocks. As such, we afford some number of channels $b_i \leq c_i$ within each block (as the experiments show, this is not really effective and should be skipped). Lastly, since we are dealing with ResNeXt blocks, we also need to pick the group width for grouped convolutions. 

This seemily generic design provides us with the following choices: we can set the number of channels $c_0, \ldots c_4$, the number of blocks per stage $d_1, \ldots d_4$, the size of the bottlenecks $b_1, \ldots b_4$, and the group widths $g_1, \ldots g_4$. In total this adds up to 17 parameters (the original paper erroneously omits to account for $c_0$) and with it, an unreasonably large number of configurations that would warrant exploring. We need some tools to reduce this huge design space effectively. Before we do so, let's implement the generic design first. 

```{.python .input  n=2}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx, init
from mxnet.gluon import nn
npx.set_np()

class AnyNet(d2l.Classifier):
    def stem(self, num_channels):
        net = nn.Sequential()
        net.add(nn.Conv2D(num_channels, kernel_size=3, padding=1, strides=2),
                nn.BatchNorm(), nn.Activation('relu'))
        return net
```

```{.python .input  n=3}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F

class AnyNet(d2l.Classifier):
    def stem(self, num_channels):
        return nn.Sequential(
            nn.LazyConv2d(num_channels, kernel_size=3, stride=2, padding=1),
            nn.LazyBatchNorm2d(), nn.ReLU())
```

```{.python .input  n=4}
%%tab tensorflow
import tensorflow as tf
from d2l import tensorflow as d2l

class AnyNet(d2l.Classifier):
    def stem(self, num_channels):
        return tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(num_channels, kernel_size=3, strides=2,
                                   padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu')])
```

Each stage consists of `depth` ResNeXt blocks,
where `num_channels` specifies the block width.
Note that the first block halves the height and width of input images.

```{.python .input  n=5}
%%tab mxnet
@d2l.add_to_class(AnyNet)
def stage(self, depth, num_channels, groups, bot_mul):
    net = nn.Sequential()
    for i in range(depth):
        if i == 0:
            net.add(d2l.ResNeXtBlock(
                num_channels, groups, bot_mul, use_1x1conv=True, strides=2))
        else:
            net.add(d2l.ResNeXtBlock(
                num_channels, num_channels, groups, bot_mul))
    return net
```

```{.python .input  n=6}
%%tab pytorch
@d2l.add_to_class(AnyNet)
def stage(self, depth, num_channels, groups, bot_mul):
    blk = []
    for i in range(depth):
        if i == 0:
            blk.append(d2l.ResNeXtBlock(num_channels, groups, bot_mul,
                use_1x1conv=True, strides=2))
        else:
            blk.append(d2l.ResNeXtBlock(num_channels, groups, bot_mul))
    return nn.Sequential(*blk)
```

```{.python .input  n=7}
%%tab tensorflow
@d2l.add_to_class(AnyNet)
def stage(self, depth, num_channels, groups, bot_mul):
    net = tf.keras.models.Sequential()
    for i in range(depth):
        if i == 0:
            net.add(d2l.ResNeXtBlock(num_channels, groups, bot_mul,
                use_1x1conv=True, strides=2))
        else:
            net.add(d2l.ResNeXtBlock(num_channels, groups, bot_mul))
    return net
```

Putting the network stem, body, and head together,
we complete the implementation of AnyNet.

```{.python .input  n=8}
%%tab all
@d2l.add_to_class(AnyNet)
def __init__(self, arch, stem_channels, lr=0.1, num_classes=10):
    super(AnyNet, self).__init__()
    self.save_hyperparameters()
    if tab.selected('mxnet'):
        self.net = nn.Sequential()
        self.net.add(self.stem(stem_channels))
        for i, s in enumerate(arch):
            self.net.add(self.stage(*s))
        self.net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
        self.net.initialize(init.Xavier())
    if tab.selected('pytorch'):
        self.net = nn.Sequential(self.stem(stem_channels))
        for i, s in enumerate(arch):
            self.net.add_module(f'stage{i+1}', self.stage(*s))
        self.net.add_module('head', nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes)))
        self.net.apply(d2l.init_cnn)
    if tab.selected('tensorflow'):
        self.net = tf.keras.models.Sequential(self.stem(stem_channels))
        for i, s in enumerate(arch):
            self.net.add(self.stage(*s))
        self.net.add(tf.keras.models.Sequential([
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Dense(units=num_classes)]))
```

## Distributions and Parameters

Consider the problem of identifying good parameters in AnyNet. We could try finding the *single best* parameter choice for a given amount of computation (FLOPs, compute time). An alternative would be to try to determine general guidelines of how the choices of parameters should be related (e.g., the size of the bottleneck, the number of channels, the number of blocks, groups). The approach in :cite:`radosavovic2019network` relies on the following four assumptions:

1. The approach assumes that general design principles can be found, such that many networks satisfying these requirements should offer good performance. Consequently, identifying a *distribution* over networks can be a good strategy. 
2. We need not train networks to convergence before we can assess whether a network is good. Instead, it is sufficient to use the intermediate results as reliable guidance for final accuracy. Using (approximate) proxies to optimize an objective is referred-to as multi-fidelity optimization :cite:`forrester2007multi`. Consequently, design optimization is carried out, based on the accuracy achieved after only a few passes through the dataset, reducing the cost significantly. 
3. Results obtained at a smaller scale (for smaller networks) generalize to larger ones. Conseqently, optimization is carried out for networks with a smaller number of blocks, fewer channels, etc.; Only in the end will we need to verify that the so-found networks also offer good performance at scale. 
4. Aspects of the design can be approximately factorized such that it is possible to infer their effect on the quality of the outcome somewhat independently. In other words, the optimization problem is moderately easy. 

These assumptions allow us to identify good network designs as follows: sample from the space of configurations uniformly and train them for a brief period of time. In particular, pick *small* networks that are relatively cheap to train, compared to a large and complex network. Given that, we can study the *distribution* of error/accuracy that can be achieved with networks that are drawn according to a given set of constraints (if any) on parameters. Denote by $F(e)$ the cumulative distribution function for errors committed by networks of a given family. That is, 

$$F(e, p) := \Pr_{\mathrm{net} \sim p(\mathrm{net})} \{e(\mathrm{net}) \leq e\}$$

Our goal is now to find a distribution $p$ over *networks* such that most networks have a very low error rate. Of course, this is computationally infeasible to perform accurately. Hence we draw a sample of networks $Z := \{\mathrm{net}_1, \ldots \mathrm{net}_n\}$ from the distribution $p$ over networks and use the empirical CDF $\hat{F}(e, Z)$ instead:

$$\hat{F}(e, Z) = \frac{1}{n}\sum_{i=1}^n \mathbf{1}(e_i \leq e)$$

The first step towards identifying a good distribution over network designs is to constrain the space we draw from. 
:citet:`Radosavovic.Kosaraju.Girshick.ea.2020` experiment with a shared network bottleneck ratio $b_i = b c_i$ for all stages $i$ of the network. This gets rid of $3$ of the $4$ parameters governing the bottleneck ratio. To assess whether this (negatively) affects the performance one can draw networks from the constrained and from the unconstrained distribution and compare the corresonding CDFs. It turns out that this constraint doesn't affect accuracy of the distribution of networks at all, as can be seen in the left panel of :numref:`fig_regnet-paper-fig5`. 
Likewise, we could choose to pick the same number *width* for all groups occurring at the various stages of the network. Again, this doesn't affect performance, as can be seen in the right panel of :numref:`fig_regnet-paper-fig5`.
Both steps combined reduce the number of free parameters by $6$. 

![Comparing error empirical distribution functions of design spaces. $\mathrm{AnyNet}_A$ is the original design, $\mathrm{AnyNet}_B$ ties the bottleneck ratios, and $\mathrm{AnyNet}_C$ also ties group widths. Left: we can see that both $A$ and $B$ perform essentially the same. Right: likewise, $B$ and $C$ perform essentially the same (Figure courtesy of :citet:`Radosavovic.Kosaraju.Girshick.ea.2020`).](../img/regnet-paper-fig5.png)
:width:`500px`
:label:`fig_regnet-paper-fig5`

Next we look for ways to reduce the multitude of potential choices for width and depth of the stages. It is a reasonable assumption that as we go deeper, the number of channels should increase, i.e., $c_i \geq c_{i-1}$. Likewise, it is equally reasonable to assume that as the stages progress, they should become deeper, i.e., $d_i \geq d_{i-1}$. 




Investigating good and bad models from $\text{AnyNetX}_C$ suggests that it may be useful to increase width across stages :cite:`Radosavovic.Kosaraju.Girshick.ea.2020`.
Empirically, simplifying
$\text{AnyNetX}_C$ to $\text{AnyNetX}_D$
with $w_{i} \leq w_{i+1}$
improves the quality of design spaces (left of  :numref:`fig_regnet-paper-fig7`).
Similarly,
adding further constraints of $d_{i} \leq d_{i+1}$
to increase network depth across stages
gives an even better $\text{AnyNetX}_E$
(right of :numref:`fig_regnet-paper-fig7`).

![Comparing error empirical distribution functions of design spaces. The legends show the min error and mean error. Increasing network width across stages (from $\text{AnyNetX}_C$ to  $\text{AnyNetX}_D$) and increasing network depth across stages (from $\text{AnyNetX}_D$ to $\text{AnyNetX}_E$) simplify the design space with improved  error distributions (figure taken from :citet:`Radosavovic.Kosaraju.Girshick.ea.2020`).](../img/regnet-paper-fig7.png)
:width:`500px`
:label:`fig_regnet-paper-fig7`



## RegNet

The resulting $\text{AnyNetX}_E$ design space
consists of simple networks
following easy-to-interpret design principles:

* Share the bottle network ratio $b_i = b$ for all stages $i$;
* Share the number of groups $g_i = g$ for all stages $i$;
* Increase network width across stages: $w_{i} \leq w_{i+1}$;
* Increase network depth across stages: $d_{i} \leq d_{i+1}$.

Following these design principles, :citet:`Radosavovic.Kosaraju.Girshick.ea.2020` proposed quantized linear constraints to
$w_i$ and $d_i$ increasing,
leading to
RegNetX using ResNeXt blocks
and RegNetY that additionally uses operators from SENets :cite:`Hu.Shen.Sun.2018`.
As an example,
we implement a 32-layer RegNetX variant
characterized by

* $b_i = 1;$
* $g_i = 16;$
* $w_1 = 32, w_2=80;$
* $d_1 = 4, d_2=6.$

```{.python .input  n=9}
%%tab all
class RegNet32(AnyNet):
    def __init__(self, lr=0.1, num_classes=10):
        stem_channels, groups, bot_mul = 32, 16, 1
        depths, channels = (4, 6), (32, 80)
        super().__init__(
            ((depths[0], channels[0], groups, bot_mul),
             (depths[1], channels[1], groups, bot_mul)),
            stem_channels, lr, num_classes)
```

We can see that each RegNet stage progressively reduces resolution and increases output channels.

```{.python .input  n=10}
%%tab mxnet, pytorch
RegNet32().layer_summary((1, 1, 96, 96))
```

```{.python .input  n=11}
%%tab tensorflow
RegNet32().layer_summary((1, 96, 96, 1))
```

## Training

Training the 32-layer RegNet on the Fashion-MNIST dataset is just like before.

```{.python .input  n=12}
%%tab mxnet, pytorch
model = RegNet32(lr=0.05)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
trainer.fit(model, data)
```

```{.python .input  n=13}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
with d2l.try_gpu():
    model = RegNet32(lr=0.01)
    trainer.fit(model, data)
```

## Discussion

With desirable properties like locality and translation invariance (:numref:`sec_why-conv`)
for vision,
CNNs have been the dominant architectures in this area.
Recently,
Transformers (:numref:`sec_transformer`) :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021,touvron2021training`
and MLPs :cite:`tolstikhin2021mlp`
have also sparked research beyond
the well-established CNN architectures for vision.
Specifically,
although lacking of the aforementioned
inductive biases inherent to CNNs,
vision Transformers (:numref:`sec_vision-transformer`)
attained state-of-the-art performance
in large-scale image classification in early 2020s,
showing that
*scalability trumps inductive biases*
:cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`.
In other words,
it is often possible to
train large Transformers
to outperform large CNNs on large datasets.
Inspired
by the superior scaling behavior of
Transformers (:numref:`sec_large-pretraining-transformers`) with multi-head self-attention (:numref:`sec_multihead-attention`),
the process of gradually
improving from a standard ResNet architecture
toward the design of a vision Transformer
leads to a family of CNNs called the ConvNeXt models
that compete favorably with Transformers for vision :cite:`liu2022convnet`.
We refer the interested readers
to CNN design discussions
in the ConvNeXt paper :cite:`liu2022convnet`.



## Exercises

1. Increase the number of stages to 4. Can you design a deeper RegNet that performs better?
1. De-ResNeXt-ify RegNets by replacing the ResNeXt block with the ResNet block. How does your new model perform?
1. Implement multiple instances of a "VioNet" family by *violating* the design principles of RegNet. How do they perform? Which of ($d_i$, $w_i$, $g_i$, $b_i$) is the most important factor?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/7462)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/7463)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/8738)
:end_tab:
