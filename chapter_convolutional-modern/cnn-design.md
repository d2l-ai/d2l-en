```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Designing Convolution Network Architectures
:label:`sec_cnn-design`

The past sections took us on a tour of modern network design for computer vision. Common to all the work we covered was that it heavily relied on the intuition of scientists. Many of the architectures are heavily informed by human creativity and to a much lesser extent by systematic exploration of the design space that deep networks offer. Nonetheless, this *network engineering* approach has been tremendously successful. 

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
ResNets (:numref:`sec_resnet`) 
changed the inductive bias towards the identity mapping (from $f(x) = 0$). This allowed for very deep networks. Almost a decade later, the ResNet design is still popular, a testament to its design. Lastly, ResNeXt (:numref:`subsec_resnext`) added grouped convolutions, offering a better trade-off between parameters and computation. A precursor to Transformers for vision, the Squeeze-and-Excitation Networks (SENets) allow for efficient information transfer between locations
:cite:`Hu.Shen.Sun.2018`. They accomplished this by computing a per-channel global attention function. 

So far we omitted networks obtained via *neural architecture search* (NAS) :cite:`zoph2016neural,liu2018darts`. We chose to do so since their cost is usually enormous, relying on brute force search, genetic algorithms, reinforcement learning, or some other form of hyperparameter optimization. Given a fixed search space,
NAS uses a search strategy to automatically select
an architecture based on the returned performance estimation.
The outcome of NAS
is a single network instance. EfficientNets are a notable outcome of this search :cite:`tan2019efficientnet`.

In the following we discuss an idea that is quite different to the quest for the *single best network*. It is computationally relatively inexpensive, it leads to scientific insights on the way, and it is quite effective in terms of the quality of outcomes. Let's review the strategy by :citet:`Radosavovic.Kosaraju.Girshick.ea.2020` to *design network design spaces*. The strategy combines the strength of manual design and NAS. It accomplishes this by operating on *distributions of networks* and optimizing the distributions in a way to obtain good performance for entire families of networks. The outcome of it are *RegNets*, specifically RegNetX and RegNetY, plus a range of guiding principles for the design of performant CNNs. 

## The AnyNet Design Space

The description below closely follows the reasoning in :citet:`Radosavovic.Kosaraju.Girshick.ea.2020` with some abbreviations to make it fit in the scope of the book. 
To begin, we need a template for the family of networks to explore. One of the commonalities of the designs in this chapter is that the networks consist of a *stem*, a *body* and a *head*. The stem performs initial image processing, often through convolutions with a larger window size. The body consists of multiple blocks, carrying out the bulk of the transformations needed to go from raw images to object representations. Lastly, the head converts this into the desired outputs, such as via a softmax regressor for multiclass classification. 
The body, in turn, consists of multiple stages, operating on the image at decreasing resolutions. In fact, both the stem and each subsequent stage quarter the spatial resolution. Lastly, each stage consists of one or more blocks. This pattern is common to all networks, from VGG to ResNeXt. Indeed, for the design of generic AnyNet networks, :citet:`Radosavovic.Kosaraju.Girshick.ea.2020` used the ResNeXt block of :numref:`fig_resnext_block`. 

![The AnyNet design. The numbers $(c,r)$ at the top of each module indicate the number of channels $c$ and the resolution $r \times r$ of the images at that point. From left to right:  generic network structure, composed of stem, body and head; 
body composed of multiple stages; detailed structure of a stage; To the right we see two alternative structures for blocks, one without downsampling and one that halves the resolution in each dimension.](../img/anynet.svg)
:label:`fig_anynet_full`

Let's review the structure outlined in :numref:`fig_anynet_full` in detail. As mentioned, an AnyNet consists of a stem, body, and head. The stem takes as its input RGB images (3 channels), using a convolution with a stride of $2$, followed by a batch norm, to halve the resolution from $r \times r$ to $r/2 \times r/2$. Moreover, it generates $c_0$ channels that serve as input to the body. 

Since the network is designed to work well with ImageNet images of $224 \times 224 \times 3$ resolution, the body serves to reduce this to $7 \times 7 \times c_4$ through 4 stages (recall that $224 = 7 \cdot 2^{1+4}$), each with an eventual stride of $2$. Lastly, the head employs an entirely standard design via global average pooling, similar to NiN :numref:`sec_nin`, followed by a fully connected layer to emit an $n$-dimensional vector for $n$-class classification. 

Most of the relevant design decisions are inherent to the body of the network. It proceeds in stages, where each stage is composed of the same type of ResNeXt blocks as we discussed in :numref:`sec_resnet`. The design there is again entirely generic: we begin with a block that halves the resolution by using a stride of $2$ (the rightmost network design). To match this, the residual branch of the ResNeXt block needs to pass through a $1 \times 1$ convolution. This block is followed by a variable number of additional ResNeXt blocks that leave both resolution and number of channels unchanged. Note that a common design practice is to add a slight bottleneck in the design of convolutional blocks. As such, we afford some number of channels $b_i \leq c_i$ within each block (as the experiments show, this is not really effective and should be skipped). Lastly, since we are dealing with ResNeXt blocks, we also need to pick the group width for grouped convolutions. 

This seemingly generic design provides us nonetheless with many choices: we can set the number of channels $c_0, \ldots c_4$, the number of blocks per stage $d_1, \ldots d_4$, the size of the bottlenecks $b_1, \ldots b_4$, and the group widths $g_1, \ldots g_4$. In total this adds up to 17 parameters and with it, an unreasonably large number of configurations that would warrant exploring. We need some tools to reduce this huge design space effectively. This is where the conceptual beauty of design spaces comes in. Before we do so, let's implement the generic design first.

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

Consider the problem of identifying good parameters in AnyNet. We could try finding the *single best* parameter choice for a given amount of computation (FLOPs, compute time). If we allowed for even only *two* possible choices for each parameter, we would have to explore $2^{17} = 131072$ combinations to find the best solution. This is clearly infeasible due to its exorbitant cost. Even worse, we don't really learn anything from this exercise in terms of how one should design a network. Next time we add, say, an SE-stage, or a shift operation, or similar, we would need to start from scratch. Even worse, due to the stochasticity in training (rounding, shuffling, bit errors), no two runs are likely to produce exactly the same results. A better strategy is to try to determine general guidelines of how the choices of parameters should be related. For instance, the size of the bottleneck, the number of channels, the number of blocks, groups, or their change between layers should ideally be governed by a collection of simple rules. The approach in :cite:`radosavovic2019network` relies on the following four assumptions:

1. We assume that general design principles actually exist, such that many networks satisfying these requirements should offer good performance. Consequently, identifying a *distribution* over networks can be a good strategy. In other words, we assume that there are many good needles in the haystack.
2. We need not train networks to convergence before we can assess whether a network is good. Instead, it is sufficient to use the intermediate results as reliable guidance for final accuracy. Using (approximate) proxies to optimize an objective is referred-to as multi-fidelity optimization :cite:`forrester2007multi`. Consequently, design optimization is carried out, based on the accuracy achieved after only a few passes through the dataset, reducing the cost significantly. 
3. Results obtained at a smaller scale (for smaller networks) generalize to larger ones. Conseqently, optimization is carried out for networks that are structurally similar, but with a smaller number of blocks, fewer channels, etc.; Only in the end will we need to verify that the so-found networks also offer good performance at scale. 
4. Aspects of the design can be approximately factorized such that it is possible to infer their effect on the quality of the outcome somewhat independently. In other words, the optimization problem is moderately easy. 

These assumptions allow us to test many networks cheaply. In particular, we can *sample* uniformly from the space of configurations and evaluate their performance. Subsequently, we can evaluate the quality of the choice of parameters by reviewing the *distribution* of error/accuracy that can be achieved with said networks. Denote by $F(e)$ the cumulative distribution function for errors committed by networks of a given family, drawn using probability disribution $p(\mathrm{net})$. That is, 

$$F(e, p) := \Pr_{\mathrm{net} \sim p(\mathrm{net})} \{e(\mathrm{net}) \leq e\}$$

Our goal is now to find a distribution $p$ over *networks* such that most networks have a very low error rate and where the support of $p(\mathrm{net})$ is concise. Of course, this is computationally infeasible to perform accurately. We resort to a sample of networks $Z := \{\mathrm{net}_1, \ldots \mathrm{net}_n\}$ from $p$ and use the empirical CDF $\hat{F}(e, Z)$ instead:

$$\hat{F}(e, Z) = \frac{1}{n}\sum_{i=1}^n \mathbf{1}(e_i \leq e)$$

Whenever the cumulative distribution function for one set of choices majorizes (or matches) another CDF it follows that its choice of parameters is superior (or indifferent). Accordingly 
:citet:`Radosavovic.Kosaraju.Girshick.ea.2020` experiment with a shared network bottleneck ratio $b_i = b c_i$ for all stages $i$ of the network. This gets rid of $3$ of the $4$ parameters governing the bottleneck ratio. To assess whether this (negatively) affects the performance one can draw networks from the constrained and from the unconstrained distribution and compare the corresonding CDFs. It turns out that this constraint doesn't affect accuracy of the distribution of networks at all, as can be seen in the first panel of :numref:`fig_regnet-fig`. 
Likewise, we could choose to pick the same number *width* for all groups occurring at the various stages of the network. Again, this doesn't affect performance, as can be seen in the second panel of :numref:`fig_regnet-fig`.
Both steps combined reduce the number of free parameters by $6$. 

![Comparing error empirical distribution functions of design spaces. $\mathrm{AnyNet}_A$ is the original design; $\mathrm{AnyNet}_B$ ties the bottleneck ratios, $\mathrm{AnyNet}_C$ also ties group widths, $\mathrm{AnyNet}_D$ strictly increases the network depth across stages. From left to right: 1) tying bottleneck ratios has no effect on performance, 2) tying group widths has no effect on performance, 3) increasing network widths (channels) across stages improves performance, 4) increasing network depths across stages improves performance. Figure courtesy of :citet:`Radosavovic.Kosaraju.Girshick.ea.2020`.](../img/regnet-fig.png)
:label:`fig_regnet-fig`

Next we look for ways to reduce the multitude of potential choices for width and depth of the stages. It is a reasonable assumption that as we go deeper, the number of channels should increase, i.e., $c_i \geq c_{i-1}$, yielding 
$\text{AnyNetX}_D$. Likewise, it is equally reasonable to assume that as the stages progress, they should become deeper, i.e., $d_i \geq d_{i-1}$, yielding $\text{AnyNetX}_E$. This can be experimentally verified in the third and fourth panel of :numref:`fig_regnet-fig`, respectively. 

## RegNet

The resulting $\text{AnyNetX}_E$ design space consists of simple networks
following easy-to-interpret design principles:

* Share the bottleneck ratio $b_i = b$ for all stages $i$;
* Share the size of the groups $g_i = g$ for all stages $i$;
* Increase network width across stages: $w_{i} \leq w_{i+1}$;
* Increase network depth across stages: $d_{i} \leq d_{i+1}$.

This leaves us with the last set of choices: how to pick the specific values for the group sizes and the widths and depths of the stages. By studying the best-performing networks from the distribution in $\text{AnyNetX}_E$ one can observe that: the width of the network ideally increases linearly with the number of blocks, i.e.\ $w_i \approx w_0 + w_a j$ where $j$ indicates the number of blocks. Given that we get to choose a different block width only per stage, we arrive at a piecewise constant function, engineered to match this dependence. Secondly, experiments also show that a bottleneck ratio of $b = 1$ performs best, i.e., we are well advised not to use bottlenecks at all! 

We recommend the interested reader to review further details for how to design specific networks for different amounts of computation by perusing :citet:`Radosavovic.Kosaraju.Girshick.ea.2020`. For instance, an effective 32-layer RegNetX variant is given by $b_i = 1$ (no bottleneck), $g_i = 16$ (group width is 16), $w_1 = 32$ and $w_2 = 80$ channels for the first and second stage respectively, chosen to be $d_1=4$ and $d_2=6$ blocks deep. The astonishing insight from the design is that it applies, even when investigating networks at larger scale. Even better, it even holds for Squeeze-Excite (SE) network designs that have a global channel activation :cite:`Hu.Shen.Sun.2018`.

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
for vision, CNNs have been the dominant architectures in this area. This has remained the case since LeNet up until recently when Transformers (:numref:`sec_transformer`) :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021,touvron2021training` started surpassing CNNs in terms of accuracy. While much of the recent progress in terms of Vision Transformers *can* be backported into CNNs :cite:`liu2022convnet`, it is only possible at a higher computational cost. Just as importantly, recent hardware optimizations (NVIDIA Ampere and Hopper) have only widened the gap in favor of transformers. 

It is worth noting that transformers have a significantly lower degree of inductive bias towards translation invariance and locality than CNNs. It is not the least due to the availability of large image collections, such as LAION-400m and LAION-5B :cite:`schuhmannlaion` with up to 5 Billion images that learned structures prevailed. Quite surprisingly, some of the more relevant work in this context even includes MLPs :cite:`tolstikhin2021mlp`. 

In sum, Vision Transformers (:numref:`sec_vision-transformer`) by now lead in terms of 
state-of-the-art performance in large-scale image classification, 
showing that *scalability trumps inductive biases* :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`.
This includes pretraining large-scale Transformers (:numref:`sec_large-pretraining-transformers`) with multi-head self-attention (:numref:`sec_multihead-attention`). We invite the readers to dive into these chapters for a much more detailed discussion.

## Exercises

1. Increase the number of stages to 4. Can you design a deeper RegNet that performs better?
1. De-ResNeXt-ify RegNets by replacing the ResNeXt block with the ResNet block. How does your new model perform?
1. Implement multiple instances of a "VioNet" family by *violating* the design principles of RegNet. How do they perform? Which of ($d_i$, $w_i$, $g_i$, $b_i$) is the most important factor?
1. Your goal is to design the 'perfect' MLP. Can you use the design principles introduced above to find good architectures? Is it possible to extrapolate from small to larger networks?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/7462)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/7463)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/8738)
:end_tab:
