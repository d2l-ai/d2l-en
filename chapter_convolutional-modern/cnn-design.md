```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch'])
```

```{.json .output n=1}
[
 {
  "data": {
   "application/vnd.jupyter.widget-view+json": {
    "model_id": "381d71084560488bbd02f19d35577d66",
    "version_major": 2,
    "version_minor": 0
   },
   "text/plain": "interactive(children=(Dropdown(description='tab', options=('mxnet', 'pytorch'), value='mxnet'), Output()), _do\u2026"
  },
  "metadata": {},
  "output_type": "display_data"
 }
]
```

# Designing Convolution Network Architectures
:label:`sec_cnn-design`

The last decade has witnessed shift
from *feature engineering* to *network engineering*
in computer vision.
Since AlexNet (:numref:`sec_alexnet`)
beat conventional computer vision models on ImageNet,
constructing very deep networks
by stacking the same blocks,
especially $3 \times 3$ convolutions,
has been popularized by VGG networks (:numref:`sec_vgg`).
The network in network (:numref:`sec_nin`)
adds local nonlinearities via $1 \times 1$ convolutions
and uses global average pooling
to aggregate information
across all locations.
GoogLeNet (:numref:`sec_googlenet`) 
is a multi-branch network that
combines the advantages from the
VGG network
and the network in network,
where its Inception block
adopts the strategy of
concatenated parallel transformations.
ResNets (:numref:`sec_resnet`)
stack residual blocks,
which are two-branch subnetworks
using identity mapping in one branch.
DenseNets (:numref:`sec_densenet`)
generalizes the residual architectures.
Other notable architectures 
include
MobileNets that use network learning to achieve high accuracy in
resource-constrained settings :cite:`Howard.Sandler.Chu.ea.2019`,
the Squeeze-and-Excitation Networks (SENets) that
allow for efficient information transfer between channels
:cite:`Hu.Shen.Sun.2018`,
and EfficientNets :cite:`tan2019efficientnet`
that scale up networks via neural architecture search.

Specifically, *neural architecture search* (NAS) :cite:`zoph2016neural,liu2018darts`
is the process of automating neural network architectures.
Given a fixed search space,
NAS uses a search strategy
to automatically select
an architecture within the search space
based on the returned performance estimation.
The outcome of NAS
is a single network instance.

Instead of focusing on designing such individual instances,
an alternative approach
is to *design network design spaces*
that characterize populations of networks :cite:`Radosavovic.Kosaraju.Girshick.ea.2020`.
This method 
combines the strength of manual design and NAS.
Through semi-automatic procedures (like in NAS),
designing network design spaces
explores the structure aspect of network design
from the initial *AnyNet* design space.
It then proceeds to discover design principles (like in manual design)
that lead to simple and regular networks: *RegNets*.
Before shedding light on these design principles,
we need to define 
the initial AnyNet design space. 
It starts with networks with
standard, fixed network blocks:
ResNeXt blocks.

## ResNeXt Blocks

ResNeXt blocks extend the residual block design (:numref:`subsec_residual-blks`)
by adding
concatenated parallel transformations
:cite:`Xie.Girshick.Dollar.ea.2017`.
Different from various transformations
in multi-branch Inception blocks,
ResNeXt adopts the same transformation in all branches,
thus minimizing manual design efforts in each branch.

![The ResNeXt block. It is a bottleneck ($b < c$) residual block with group convolution ($g$ groups).](../img/resnext-block.svg)
:label:`fig_resnext_block`

The left dotted box in
:numref:`fig_resnext_block`
depicts the added concatenated parallel transformation
strategy in ResNeXt.
More concretely,
an input with $c$ channels
is first split into $g$ groups
via $g$ branches of $1 \times 1$ convolutions
followed by $3 \times 3$ convolutions,
all with $b/g$ output channels.
Concatenating these $g$ outputs
results in $b$ output channels,
leading to "bottlenecked" (when $b < c$) network width
inside the dashed box.
This output
will restore the original $c$ channels of the input
via the final $1 \times 1$ convolution
right before summing with the residual connection.
Notably,
the left dotted box is equivalent to 
the much *simplified* right dotted box in :numref:`fig_resnext_block`,
where we only need to specify 
that the $3 \times 3$ convolution is a *group convolution*
with $g$ groups.
The group convolution dates back 
to the idea of distributing the AlexNet
model over two GPUs :cite:`Krizhevsky.Sutskever.Hinton.2012`.

The following implementation of the `ResNeXtBlock` class
treats `group_width` ($b/g$ in :numref:`fig_resnext_block`) as an argument
so that given `bot_channels` ($b$ in :numref:`fig_resnext_block`) bottleneck channels,
the $3 \times 3$ group convolution will
have `bot_channels//group_width` groups.
Similar to
the residual block implementation in
:numref:`subsec_residual-blks`,
the residual connection
is generalized
with a $1 \times 1$ convolution (`conv4`),
where setting `use_1x1conv=True, strides=2`
halves the output height and width.

```{.python .input  n=2}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx, init
from mxnet.gluon import nn
npx.set_np()

class ResNeXtBlock(nn.Block):
    """The ResNeXt block."""
    def __init__(self, num_channels, group_width, bot_mul,
                 use_1x1conv=False, strides=1, **kwargs):
        super().__init__(**kwargs)
        bot_channels = int(round(num_channels * bot_mul))
        self.conv1 = nn.Conv2D(bot_channels, kernel_size=1, padding=0,
                               strides=1)
        self.conv2 = nn.Conv2D(bot_channels, kernel_size=3, padding=1,
                               strides=strides,
                               groups=bot_channels//group_width)
        self.conv3 = nn.Conv2D(num_channels, kernel_size=1, padding=0,
                               strides=1)
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()
        self.bn3 = nn.BatchNorm()
        if use_1x1conv:
            self.conv4 = nn.Conv2D(num_channels, kernel_size=1,
                                   strides=strides)
            self.bn4 = nn.BatchNorm()
        else:
            self.conv4 = None
        
    def forward(self, X):
        Y = npx.relu(self.bn1(self.conv1(X)))
        Y = npx.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return npx.relu(Y + X)
```

```{.python .input  n=3}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F

class ResNeXtBlock(nn.Module):
    """The ResNeXt block."""
    def __init__(self, input_channels, num_channels, group_width, bot_mul,
                 use_1x1conv=False, strides=1):
        super().__init__()
        bot_channels = int(round(num_channels * bot_mul))
        self.conv1 = nn.Conv2d(input_channels, bot_channels, kernel_size=1,
                               stride=1)
        self.conv2 = nn.Conv2d(bot_channels, bot_channels, kernel_size=3,
                               stride=strides, padding=1,
                               groups=bot_channels//group_width)
        self.conv3 = nn.Conv2d(bot_channels, num_channels, kernel_size=1,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(bot_channels)
        self.bn2 = nn.BatchNorm2d(bot_channels)
        self.bn3 = nn.BatchNorm2d(num_channels)
        if use_1x1conv:
            self.conv4 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
            self.bn4 = nn.BatchNorm2d(num_channels)
        else:
            self.conv4 = None
        
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return F.relu(Y + X)
```

```{.json .output n=3}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Ignored to run as it is not marked as a \"mxnet\" cell."
 }
]
```

In the following case (`use_1x1conv=False, strides=1`), the input and output are of the same shape.

```{.python .input  n=4}
%%tab all
if tab.selected('mxnet'):
    blk = ResNeXtBlock(32, 16, 1)
    blk.initialize()
if tab.selected('pytorch'):
    blk = ResNeXtBlock(32, 32, 16, 1)
X = d2l.randn(4, 32, 96, 96)
blk(X).shape
```

```{.json .output n=4}
[
 {
  "data": {
   "text/plain": "(4, 32, 96, 96)"
  },
  "execution_count": 4,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Alternatively, setting `use_1x1conv=True, strides=2`
halves the output height and width.

```{.python .input  n=5}
%%tab all
if tab.selected('mxnet'):
    blk = ResNeXtBlock(32, 16, 1, use_1x1conv=True, strides=2)
    blk.initialize()
if tab.selected('pytorch'):
    blk = ResNeXtBlock(32, 32, 16, 1, use_1x1conv=True, strides=2)
blk(X).shape
```

```{.json .output n=5}
[
 {
  "data": {
   "text/plain": "(4, 32, 48, 48)"
  },
  "execution_count": 5,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

A key advantage of the ResNeXt design
is that increasing groups
leads to sparser connections (i.e., lower computational complexity) within the block,
thus enabling an increase of network width
to achieve a better tradeoff between
FLOPs (floating-point operations in number of multiply-adds) and accuracy.
Thus, ResNeXt-ification
is appealing in convolution network design :cite:`liu2022convnet`
and the following AnyNet design space
will be based on the ResNeXt block.


## AnyNet

Now, we implement this module. Note that special processing has been performed on the first module.

```{.python .input  n=6}
%%tab mxnet
class AnyNet(d2l.Classifier):
    def stem(self, num_channels):
        net = nn.Sequential()
        net.add(nn.Conv2D(num_channels, kernel_size=3, padding=1, strides=2),
                nn.BatchNorm(), nn.Activation('relu'))
        return net
```

```{.python .input  n=7}
%%tab pytorch
class AnyNet(d2l.Classifier):
    def stem(self, input_channels, num_channels):
        return nn.Sequential(
            nn.Conv2d(input_channels, num_channels, kernel_size=3, stride=2,
                      padding=1),
            nn.BatchNorm2d(num_channels), nn.ReLU())
```

```{.json .output n=7}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Ignored to run as it is not marked as a \"mxnet\" cell."
 }
]
```

other block

```{.python .input  n=8}
%%tab mxnet
@d2l.add_to_class(AnyNet)
def stage(self, depth, num_channels, group_width, bot_mul):
    net = nn.Sequential()
    for i in range(depth):
        if i == 0:
            net.add(ResNeXtBlock(
                num_channels, group_width, bot_mul, use_1x1conv=True,
                strides=2))
        else:
            net.add(ResNeXtBlock(
                num_channels, num_channels, group_width, bot_mul))
    return net
```

```{.python .input  n=9}
%%tab pytorch
@d2l.add_to_class(AnyNet)
def stage(self, depth, input_channels, num_channels, group_width, bot_mul):
    blk = []
    for i in range(depth):
        if i == 0:
            blk.append(ResNeXtBlock(
                input_channels, num_channels, group_width, bot_mul,
                use_1x1conv=True, strides=2))
        else:
            blk.append(ResNeXtBlock(
                num_channels, num_channels, group_width, bot_mul))
    return nn.Sequential(*blk)
```

```{.json .output n=9}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Ignored to run as it is not marked as a \"mxnet\" cell."
 }
]
```

Then, we add all the modules to RegNet.

```{.python .input  n=10}
%%tab all
@d2l.add_to_class(AnyNet)
def __init__(self, arch, stem_channels, num_classes=10, lr=0.1):
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
        self.net = nn.Sequential(self.stem(1, stem_channels))
        for i, s in enumerate(arch):
            self.net.add_module(f'stage{i+1}', self.stage(*s))
        self.net.add_module('head', nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(arch[-1][2], num_classes)))
        self.net.apply(d2l.init_cnn_weights)
```

## RegNet

* Design choice in AnyNet
* Design principles for RegNet
* AnyNet -> RegNet


Before training RegNet, let's observe how the input shape changes across different modules in ResNet.

```{.python .input  n=11}
%%tab all
class RegNet32(AnyNet):
    def __init__(self, num_classes=10, lr=0.1):
        stem_channels, group_width, bot_mul = 32, 16, 1
        depths, channels = [4, 6], [32, 80]
        if tab.selected(['mxnet']):
            super().__init__(
                ((depths[0], channels[0], group_width, bot_mul),
                 (depths[1], channels[1], group_width, bot_mul)),
                stem_channels, num_classes, lr)
        if tab.selected(['pytorch']):
            super().__init__(
                ((depths[0], stem_channels, channels[0], group_width, bot_mul),
                 (depths[1], channels[0], channels[1], group_width, bot_mul)),
                stem_channels, num_classes, lr)
```

```{.python .input  n=12}
%%tab all
RegNet32().layer_summary((1, 1, 96, 96))
```

```{.json .output n=12}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Sequential output shape:\t (1, 32, 48, 48)\nSequential output shape:\t (1, 32, 24, 24)\nSequential output shape:\t (1, 80, 12, 12)\nGlobalAvgPool2D output shape:\t (1, 80, 1, 1)\nDense output shape:\t (1, 10)\n"
 }
]
```

## Training

We train RegNet on the Fashion-MNIST dataset, just like before.

```{.python .input  n=13}
%%tab all
model = RegNet32(lr=0.05)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
trainer.fit(model, data)
```

```{.json .output n=13}
[
 {
  "data": {
   "image/svg+xml": "<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"no\"?>\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n  \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n<!-- Created with matplotlib (https://matplotlib.org/) -->\n<svg height=\"180.65625pt\" version=\"1.1\" viewBox=\"0 0 238.965625 180.65625\" width=\"238.965625pt\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n <metadata>\n  <rdf:RDF xmlns:cc=\"http://creativecommons.org/ns#\" xmlns:dc=\"http://purl.org/dc/elements/1.1/\" xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n   <cc:Work>\n    <dc:type rdf:resource=\"http://purl.org/dc/dcmitype/StillImage\"/>\n    <dc:date>2022-03-06T07:31:49.402222</dc:date>\n    <dc:format>image/svg+xml</dc:format>\n    <dc:creator>\n     <cc:Agent>\n      <dc:title>Matplotlib v3.3.3, https://matplotlib.org/</dc:title>\n     </cc:Agent>\n    </dc:creator>\n   </cc:Work>\n  </rdf:RDF>\n </metadata>\n <defs>\n  <style type=\"text/css\">*{stroke-linecap:butt;stroke-linejoin:round;}</style>\n </defs>\n <g id=\"figure_1\">\n  <g id=\"patch_1\">\n   <path d=\"M 0 180.65625 \nL 238.965625 180.65625 \nL 238.965625 0 \nL 0 0 \nz\n\" style=\"fill:none;\"/>\n  </g>\n  <g id=\"axes_1\">\n   <g id=\"patch_2\">\n    <path d=\"M 30.103125 143.1 \nL 225.403125 143.1 \nL 225.403125 7.2 \nL 30.103125 7.2 \nz\n\" style=\"fill:#ffffff;\"/>\n   </g>\n   <g id=\"matplotlib.axis_1\">\n    <g id=\"xtick_1\">\n     <g id=\"line2d_1\">\n      <defs>\n       <path d=\"M 0 0 \nL 0 3.5 \n\" id=\"ma01645d79f\" style=\"stroke:#000000;stroke-width:0.8;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#ma01645d79f\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_1\">\n      <!-- 0 -->\n      <g transform=\"translate(26.921875 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 31.78125 66.40625 \nQ 24.171875 66.40625 20.328125 58.90625 \nQ 16.5 51.421875 16.5 36.375 \nQ 16.5 21.390625 20.328125 13.890625 \nQ 24.171875 6.390625 31.78125 6.390625 \nQ 39.453125 6.390625 43.28125 13.890625 \nQ 47.125 21.390625 47.125 36.375 \nQ 47.125 51.421875 43.28125 58.90625 \nQ 39.453125 66.40625 31.78125 66.40625 \nz\nM 31.78125 74.21875 \nQ 44.046875 74.21875 50.515625 64.515625 \nQ 56.984375 54.828125 56.984375 36.375 \nQ 56.984375 17.96875 50.515625 8.265625 \nQ 44.046875 -1.421875 31.78125 -1.421875 \nQ 19.53125 -1.421875 13.0625 8.265625 \nQ 6.59375 17.96875 6.59375 36.375 \nQ 6.59375 54.828125 13.0625 64.515625 \nQ 19.53125 74.21875 31.78125 74.21875 \nz\n\" id=\"DejaVuSans-48\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_2\">\n     <g id=\"line2d_2\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"69.163125\" xlink:href=\"#ma01645d79f\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_2\">\n      <!-- 2 -->\n      <g transform=\"translate(65.981875 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 19.1875 8.296875 \nL 53.609375 8.296875 \nL 53.609375 0 \nL 7.328125 0 \nL 7.328125 8.296875 \nQ 12.9375 14.109375 22.625 23.890625 \nQ 32.328125 33.6875 34.8125 36.53125 \nQ 39.546875 41.84375 41.421875 45.53125 \nQ 43.3125 49.21875 43.3125 52.78125 \nQ 43.3125 58.59375 39.234375 62.25 \nQ 35.15625 65.921875 28.609375 65.921875 \nQ 23.96875 65.921875 18.8125 64.3125 \nQ 13.671875 62.703125 7.8125 59.421875 \nL 7.8125 69.390625 \nQ 13.765625 71.78125 18.9375 73 \nQ 24.125 74.21875 28.421875 74.21875 \nQ 39.75 74.21875 46.484375 68.546875 \nQ 53.21875 62.890625 53.21875 53.421875 \nQ 53.21875 48.921875 51.53125 44.890625 \nQ 49.859375 40.875 45.40625 35.40625 \nQ 44.1875 33.984375 37.640625 27.21875 \nQ 31.109375 20.453125 19.1875 8.296875 \nz\n\" id=\"DejaVuSans-50\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-50\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_3\">\n     <g id=\"line2d_3\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"108.223125\" xlink:href=\"#ma01645d79f\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_3\">\n      <!-- 4 -->\n      <g transform=\"translate(105.041875 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 37.796875 64.3125 \nL 12.890625 25.390625 \nL 37.796875 25.390625 \nz\nM 35.203125 72.90625 \nL 47.609375 72.90625 \nL 47.609375 25.390625 \nL 58.015625 25.390625 \nL 58.015625 17.1875 \nL 47.609375 17.1875 \nL 47.609375 0 \nL 37.796875 0 \nL 37.796875 17.1875 \nL 4.890625 17.1875 \nL 4.890625 26.703125 \nz\n\" id=\"DejaVuSans-52\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-52\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_4\">\n     <g id=\"line2d_4\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"147.283125\" xlink:href=\"#ma01645d79f\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_4\">\n      <!-- 6 -->\n      <g transform=\"translate(144.101875 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 33.015625 40.375 \nQ 26.375 40.375 22.484375 35.828125 \nQ 18.609375 31.296875 18.609375 23.390625 \nQ 18.609375 15.53125 22.484375 10.953125 \nQ 26.375 6.390625 33.015625 6.390625 \nQ 39.65625 6.390625 43.53125 10.953125 \nQ 47.40625 15.53125 47.40625 23.390625 \nQ 47.40625 31.296875 43.53125 35.828125 \nQ 39.65625 40.375 33.015625 40.375 \nz\nM 52.59375 71.296875 \nL 52.59375 62.3125 \nQ 48.875 64.0625 45.09375 64.984375 \nQ 41.3125 65.921875 37.59375 65.921875 \nQ 27.828125 65.921875 22.671875 59.328125 \nQ 17.53125 52.734375 16.796875 39.40625 \nQ 19.671875 43.65625 24.015625 45.921875 \nQ 28.375 48.1875 33.59375 48.1875 \nQ 44.578125 48.1875 50.953125 41.515625 \nQ 57.328125 34.859375 57.328125 23.390625 \nQ 57.328125 12.15625 50.6875 5.359375 \nQ 44.046875 -1.421875 33.015625 -1.421875 \nQ 20.359375 -1.421875 13.671875 8.265625 \nQ 6.984375 17.96875 6.984375 36.375 \nQ 6.984375 53.65625 15.1875 63.9375 \nQ 23.390625 74.21875 37.203125 74.21875 \nQ 40.921875 74.21875 44.703125 73.484375 \nQ 48.484375 72.75 52.59375 71.296875 \nz\n\" id=\"DejaVuSans-54\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-54\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_5\">\n     <g id=\"line2d_5\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"186.343125\" xlink:href=\"#ma01645d79f\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_5\">\n      <!-- 8 -->\n      <g transform=\"translate(183.161875 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 31.78125 34.625 \nQ 24.75 34.625 20.71875 30.859375 \nQ 16.703125 27.09375 16.703125 20.515625 \nQ 16.703125 13.921875 20.71875 10.15625 \nQ 24.75 6.390625 31.78125 6.390625 \nQ 38.8125 6.390625 42.859375 10.171875 \nQ 46.921875 13.96875 46.921875 20.515625 \nQ 46.921875 27.09375 42.890625 30.859375 \nQ 38.875 34.625 31.78125 34.625 \nz\nM 21.921875 38.8125 \nQ 15.578125 40.375 12.03125 44.71875 \nQ 8.5 49.078125 8.5 55.328125 \nQ 8.5 64.0625 14.71875 69.140625 \nQ 20.953125 74.21875 31.78125 74.21875 \nQ 42.671875 74.21875 48.875 69.140625 \nQ 55.078125 64.0625 55.078125 55.328125 \nQ 55.078125 49.078125 51.53125 44.71875 \nQ 48 40.375 41.703125 38.8125 \nQ 48.828125 37.15625 52.796875 32.3125 \nQ 56.78125 27.484375 56.78125 20.515625 \nQ 56.78125 9.90625 50.3125 4.234375 \nQ 43.84375 -1.421875 31.78125 -1.421875 \nQ 19.734375 -1.421875 13.25 4.234375 \nQ 6.78125 9.90625 6.78125 20.515625 \nQ 6.78125 27.484375 10.78125 32.3125 \nQ 14.796875 37.15625 21.921875 38.8125 \nz\nM 18.3125 54.390625 \nQ 18.3125 48.734375 21.84375 45.5625 \nQ 25.390625 42.390625 31.78125 42.390625 \nQ 38.140625 42.390625 41.71875 45.5625 \nQ 45.3125 48.734375 45.3125 54.390625 \nQ 45.3125 60.0625 41.71875 63.234375 \nQ 38.140625 66.40625 31.78125 66.40625 \nQ 25.390625 66.40625 21.84375 63.234375 \nQ 18.3125 60.0625 18.3125 54.390625 \nz\n\" id=\"DejaVuSans-56\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-56\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_6\">\n     <g id=\"line2d_6\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"225.403125\" xlink:href=\"#ma01645d79f\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_6\">\n      <!-- 10 -->\n      <g transform=\"translate(219.040625 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 12.40625 8.296875 \nL 28.515625 8.296875 \nL 28.515625 63.921875 \nL 10.984375 60.40625 \nL 10.984375 69.390625 \nL 28.421875 72.90625 \nL 38.28125 72.90625 \nL 38.28125 8.296875 \nL 54.390625 8.296875 \nL 54.390625 0 \nL 12.40625 0 \nz\n\" id=\"DejaVuSans-49\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-49\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"text_7\">\n     <!-- epoch -->\n     <g transform=\"translate(112.525 171.376563)scale(0.1 -0.1)\">\n      <defs>\n       <path d=\"M 56.203125 29.59375 \nL 56.203125 25.203125 \nL 14.890625 25.203125 \nQ 15.484375 15.921875 20.484375 11.0625 \nQ 25.484375 6.203125 34.421875 6.203125 \nQ 39.59375 6.203125 44.453125 7.46875 \nQ 49.3125 8.734375 54.109375 11.28125 \nL 54.109375 2.78125 \nQ 49.265625 0.734375 44.1875 -0.34375 \nQ 39.109375 -1.421875 33.890625 -1.421875 \nQ 20.796875 -1.421875 13.15625 6.1875 \nQ 5.515625 13.8125 5.515625 26.8125 \nQ 5.515625 40.234375 12.765625 48.109375 \nQ 20.015625 56 32.328125 56 \nQ 43.359375 56 49.78125 48.890625 \nQ 56.203125 41.796875 56.203125 29.59375 \nz\nM 47.21875 32.234375 \nQ 47.125 39.59375 43.09375 43.984375 \nQ 39.0625 48.390625 32.421875 48.390625 \nQ 24.90625 48.390625 20.390625 44.140625 \nQ 15.875 39.890625 15.1875 32.171875 \nz\n\" id=\"DejaVuSans-101\"/>\n       <path d=\"M 18.109375 8.203125 \nL 18.109375 -20.796875 \nL 9.078125 -20.796875 \nL 9.078125 54.6875 \nL 18.109375 54.6875 \nL 18.109375 46.390625 \nQ 20.953125 51.265625 25.265625 53.625 \nQ 29.59375 56 35.59375 56 \nQ 45.5625 56 51.78125 48.09375 \nQ 58.015625 40.1875 58.015625 27.296875 \nQ 58.015625 14.40625 51.78125 6.484375 \nQ 45.5625 -1.421875 35.59375 -1.421875 \nQ 29.59375 -1.421875 25.265625 0.953125 \nQ 20.953125 3.328125 18.109375 8.203125 \nz\nM 48.6875 27.296875 \nQ 48.6875 37.203125 44.609375 42.84375 \nQ 40.53125 48.484375 33.40625 48.484375 \nQ 26.265625 48.484375 22.1875 42.84375 \nQ 18.109375 37.203125 18.109375 27.296875 \nQ 18.109375 17.390625 22.1875 11.75 \nQ 26.265625 6.109375 33.40625 6.109375 \nQ 40.53125 6.109375 44.609375 11.75 \nQ 48.6875 17.390625 48.6875 27.296875 \nz\n\" id=\"DejaVuSans-112\"/>\n       <path d=\"M 30.609375 48.390625 \nQ 23.390625 48.390625 19.1875 42.75 \nQ 14.984375 37.109375 14.984375 27.296875 \nQ 14.984375 17.484375 19.15625 11.84375 \nQ 23.34375 6.203125 30.609375 6.203125 \nQ 37.796875 6.203125 41.984375 11.859375 \nQ 46.1875 17.53125 46.1875 27.296875 \nQ 46.1875 37.015625 41.984375 42.703125 \nQ 37.796875 48.390625 30.609375 48.390625 \nz\nM 30.609375 56 \nQ 42.328125 56 49.015625 48.375 \nQ 55.71875 40.765625 55.71875 27.296875 \nQ 55.71875 13.875 49.015625 6.21875 \nQ 42.328125 -1.421875 30.609375 -1.421875 \nQ 18.84375 -1.421875 12.171875 6.21875 \nQ 5.515625 13.875 5.515625 27.296875 \nQ 5.515625 40.765625 12.171875 48.375 \nQ 18.84375 56 30.609375 56 \nz\n\" id=\"DejaVuSans-111\"/>\n       <path d=\"M 48.78125 52.59375 \nL 48.78125 44.1875 \nQ 44.96875 46.296875 41.140625 47.34375 \nQ 37.3125 48.390625 33.40625 48.390625 \nQ 24.65625 48.390625 19.8125 42.84375 \nQ 14.984375 37.3125 14.984375 27.296875 \nQ 14.984375 17.28125 19.8125 11.734375 \nQ 24.65625 6.203125 33.40625 6.203125 \nQ 37.3125 6.203125 41.140625 7.25 \nQ 44.96875 8.296875 48.78125 10.40625 \nL 48.78125 2.09375 \nQ 45.015625 0.34375 40.984375 -0.53125 \nQ 36.96875 -1.421875 32.421875 -1.421875 \nQ 20.0625 -1.421875 12.78125 6.34375 \nQ 5.515625 14.109375 5.515625 27.296875 \nQ 5.515625 40.671875 12.859375 48.328125 \nQ 20.21875 56 33.015625 56 \nQ 37.15625 56 41.109375 55.140625 \nQ 45.0625 54.296875 48.78125 52.59375 \nz\n\" id=\"DejaVuSans-99\"/>\n       <path d=\"M 54.890625 33.015625 \nL 54.890625 0 \nL 45.90625 0 \nL 45.90625 32.71875 \nQ 45.90625 40.484375 42.875 44.328125 \nQ 39.84375 48.1875 33.796875 48.1875 \nQ 26.515625 48.1875 22.3125 43.546875 \nQ 18.109375 38.921875 18.109375 30.90625 \nL 18.109375 0 \nL 9.078125 0 \nL 9.078125 75.984375 \nL 18.109375 75.984375 \nL 18.109375 46.1875 \nQ 21.34375 51.125 25.703125 53.5625 \nQ 30.078125 56 35.796875 56 \nQ 45.21875 56 50.046875 50.171875 \nQ 54.890625 44.34375 54.890625 33.015625 \nz\n\" id=\"DejaVuSans-104\"/>\n      </defs>\n      <use xlink:href=\"#DejaVuSans-101\"/>\n      <use x=\"61.523438\" xlink:href=\"#DejaVuSans-112\"/>\n      <use x=\"125\" xlink:href=\"#DejaVuSans-111\"/>\n      <use x=\"186.181641\" xlink:href=\"#DejaVuSans-99\"/>\n      <use x=\"241.162109\" xlink:href=\"#DejaVuSans-104\"/>\n     </g>\n    </g>\n   </g>\n   <g id=\"matplotlib.axis_2\">\n    <g id=\"ytick_1\">\n     <g id=\"line2d_7\">\n      <defs>\n       <path d=\"M 0 0 \nL -3.5 0 \n\" id=\"m2675c53ae7\" style=\"stroke:#000000;stroke-width:0.8;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#m2675c53ae7\" y=\"142.93477\"/>\n      </g>\n     </g>\n     <g id=\"text_8\">\n      <!-- 0.0 -->\n      <g transform=\"translate(7.2 146.733989)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 10.6875 12.40625 \nL 21 12.40625 \nL 21 0 \nL 10.6875 0 \nz\n\" id=\"DejaVuSans-46\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_2\">\n     <g id=\"line2d_8\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#m2675c53ae7\" y=\"114.769976\"/>\n      </g>\n     </g>\n     <g id=\"text_9\">\n      <!-- 0.2 -->\n      <g transform=\"translate(7.2 118.569195)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-50\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_3\">\n     <g id=\"line2d_9\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#m2675c53ae7\" y=\"86.605181\"/>\n      </g>\n     </g>\n     <g id=\"text_10\">\n      <!-- 0.4 -->\n      <g transform=\"translate(7.2 90.4044)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-52\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_4\">\n     <g id=\"line2d_10\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#m2675c53ae7\" y=\"58.440387\"/>\n      </g>\n     </g>\n     <g id=\"text_11\">\n      <!-- 0.6 -->\n      <g transform=\"translate(7.2 62.239606)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-54\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_5\">\n     <g id=\"line2d_11\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#m2675c53ae7\" y=\"30.275592\"/>\n      </g>\n     </g>\n     <g id=\"text_12\">\n      <!-- 0.8 -->\n      <g transform=\"translate(7.2 34.074811)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-56\"/>\n      </g>\n     </g>\n    </g>\n   </g>\n   <g id=\"line2d_12\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 34.954394 48.160338 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_13\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 34.954394 48.160338 \nL 44.698573 93.128243 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_14\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 34.954394 48.160338 \nL 44.698573 93.128243 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_15\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 95.714911 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_16\"/>\n   <g id=\"line2d_17\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 34.954394 48.160338 \nL 44.698573 93.128243 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_18\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 95.714911 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_19\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 18.878209 \n\" style=\"fill:none;stroke:#2ca02c;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_20\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 34.954394 48.160338 \nL 44.698573 93.128243 \nL 54.442752 102.486279 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_21\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 95.714911 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_22\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 18.878209 \n\" style=\"fill:none;stroke:#2ca02c;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_23\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 34.954394 48.160338 \nL 44.698573 93.128243 \nL 54.442752 102.486279 \nL 64.186931 105.517046 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_24\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 95.714911 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_25\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 18.878209 \n\" style=\"fill:none;stroke:#2ca02c;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_26\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 34.954394 48.160338 \nL 44.698573 93.128243 \nL 54.442752 102.486279 \nL 64.186931 105.517046 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_27\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 95.714911 \nL 69.163125 99.66838 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_28\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 18.878209 \n\" style=\"fill:none;stroke:#2ca02c;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_29\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 34.954394 48.160338 \nL 44.698573 93.128243 \nL 54.442752 102.486279 \nL 64.186931 105.517046 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_30\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 95.714911 \nL 69.163125 99.66838 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_31\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 18.878209 \nL 69.163125 18.488269 \n\" style=\"fill:none;stroke:#2ca02c;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_32\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 34.954394 48.160338 \nL 44.698573 93.128243 \nL 54.442752 102.486279 \nL 64.186931 105.517046 \nL 73.93111 111.715817 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_33\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 95.714911 \nL 69.163125 99.66838 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_34\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 18.878209 \nL 69.163125 18.488269 \n\" style=\"fill:none;stroke:#2ca02c;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_35\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 34.954394 48.160338 \nL 44.698573 93.128243 \nL 54.442752 102.486279 \nL 64.186931 105.517046 \nL 73.93111 111.715817 \nL 83.675289 111.435921 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_36\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 95.714911 \nL 69.163125 99.66838 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_37\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 18.878209 \nL 69.163125 18.488269 \n\" style=\"fill:none;stroke:#2ca02c;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_38\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 34.954394 48.160338 \nL 44.698573 93.128243 \nL 54.442752 102.486279 \nL 64.186931 105.517046 \nL 73.93111 111.715817 \nL 83.675289 111.435921 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_39\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 95.714911 \nL 69.163125 99.66838 \nL 88.693125 100.613576 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_40\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 18.878209 \nL 69.163125 18.488269 \n\" style=\"fill:none;stroke:#2ca02c;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_41\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 34.954394 48.160338 \nL 44.698573 93.128243 \nL 54.442752 102.486279 \nL 64.186931 105.517046 \nL 73.93111 111.715817 \nL 83.675289 111.435921 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_42\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 95.714911 \nL 69.163125 99.66838 \nL 88.693125 100.613576 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_43\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 18.878209 \nL 69.163125 18.488269 \nL 88.693125 17.039922 \n\" style=\"fill:none;stroke:#2ca02c;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_44\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 34.954394 48.160338 \nL 44.698573 93.128243 \nL 54.442752 102.486279 \nL 64.186931 105.517046 \nL 73.93111 111.715817 \nL 83.675289 111.435921 \nL 93.419468 115.799891 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_45\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 95.714911 \nL 69.163125 99.66838 \nL 88.693125 100.613576 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_46\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 18.878209 \nL 69.163125 18.488269 \nL 88.693125 17.039922 \n\" style=\"fill:none;stroke:#2ca02c;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_47\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 34.954394 48.160338 \nL 44.698573 93.128243 \nL 54.442752 102.486279 \nL 64.186931 105.517046 \nL 73.93111 111.715817 \nL 83.675289 111.435921 \nL 93.419468 115.799891 \nL 103.163647 117.005553 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_48\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 95.714911 \nL 69.163125 99.66838 \nL 88.693125 100.613576 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_49\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 18.878209 \nL 69.163125 18.488269 \nL 88.693125 17.039922 \n\" style=\"fill:none;stroke:#2ca02c;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_50\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 34.954394 48.160338 \nL 44.698573 93.128243 \nL 54.442752 102.486279 \nL 64.186931 105.517046 \nL 73.93111 111.715817 \nL 83.675289 111.435921 \nL 93.419468 115.799891 \nL 103.163647 117.005553 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_51\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 95.714911 \nL 69.163125 99.66838 \nL 88.693125 100.613576 \nL 108.223125 106.10921 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_52\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 18.878209 \nL 69.163125 18.488269 \nL 88.693125 17.039922 \n\" style=\"fill:none;stroke:#2ca02c;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_53\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 34.954394 48.160338 \nL 44.698573 93.128243 \nL 54.442752 102.486279 \nL 64.186931 105.517046 \nL 73.93111 111.715817 \nL 83.675289 111.435921 \nL 93.419468 115.799891 \nL 103.163647 117.005553 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_54\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 95.714911 \nL 69.163125 99.66838 \nL 88.693125 100.613576 \nL 108.223125 106.10921 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_55\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 18.878209 \nL 69.163125 18.488269 \nL 88.693125 17.039922 \nL 108.223125 14.714209 \n\" style=\"fill:none;stroke:#2ca02c;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_56\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 34.954394 48.160338 \nL 44.698573 93.128243 \nL 54.442752 102.486279 \nL 64.186931 105.517046 \nL 73.93111 111.715817 \nL 83.675289 111.435921 \nL 93.419468 115.799891 \nL 103.163647 117.005553 \nL 112.907826 121.163234 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_57\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 95.714911 \nL 69.163125 99.66838 \nL 88.693125 100.613576 \nL 108.223125 106.10921 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_58\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 18.878209 \nL 69.163125 18.488269 \nL 88.693125 17.039922 \nL 108.223125 14.714209 \n\" style=\"fill:none;stroke:#2ca02c;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_59\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 34.954394 48.160338 \nL 44.698573 93.128243 \nL 54.442752 102.486279 \nL 64.186931 105.517046 \nL 73.93111 111.715817 \nL 83.675289 111.435921 \nL 93.419468 115.799891 \nL 103.163647 117.005553 \nL 112.907826 121.163234 \nL 122.652006 119.718018 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_60\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 95.714911 \nL 69.163125 99.66838 \nL 88.693125 100.613576 \nL 108.223125 106.10921 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_61\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 18.878209 \nL 69.163125 18.488269 \nL 88.693125 17.039922 \nL 108.223125 14.714209 \n\" style=\"fill:none;stroke:#2ca02c;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_62\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 34.954394 48.160338 \nL 44.698573 93.128243 \nL 54.442752 102.486279 \nL 64.186931 105.517046 \nL 73.93111 111.715817 \nL 83.675289 111.435921 \nL 93.419468 115.799891 \nL 103.163647 117.005553 \nL 112.907826 121.163234 \nL 122.652006 119.718018 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_63\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 95.714911 \nL 69.163125 99.66838 \nL 88.693125 100.613576 \nL 108.223125 106.10921 \nL 127.753125 97.584144 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_64\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 18.878209 \nL 69.163125 18.488269 \nL 88.693125 17.039922 \nL 108.223125 14.714209 \n\" style=\"fill:none;stroke:#2ca02c;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_65\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 34.954394 48.160338 \nL 44.698573 93.128243 \nL 54.442752 102.486279 \nL 64.186931 105.517046 \nL 73.93111 111.715817 \nL 83.675289 111.435921 \nL 93.419468 115.799891 \nL 103.163647 117.005553 \nL 112.907826 121.163234 \nL 122.652006 119.718018 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_66\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 95.714911 \nL 69.163125 99.66838 \nL 88.693125 100.613576 \nL 108.223125 106.10921 \nL 127.753125 97.584144 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_67\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 18.878209 \nL 69.163125 18.488269 \nL 88.693125 17.039922 \nL 108.223125 14.714209 \nL 127.753125 17.207039 \n\" style=\"fill:none;stroke:#2ca02c;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_68\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 34.954394 48.160338 \nL 44.698573 93.128243 \nL 54.442752 102.486279 \nL 64.186931 105.517046 \nL 73.93111 111.715817 \nL 83.675289 111.435921 \nL 93.419468 115.799891 \nL 103.163647 117.005553 \nL 112.907826 121.163234 \nL 122.652006 119.718018 \nL 132.396185 125.335394 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_69\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 95.714911 \nL 69.163125 99.66838 \nL 88.693125 100.613576 \nL 108.223125 106.10921 \nL 127.753125 97.584144 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_70\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 18.878209 \nL 69.163125 18.488269 \nL 88.693125 17.039922 \nL 108.223125 14.714209 \nL 127.753125 17.207039 \n\" style=\"fill:none;stroke:#2ca02c;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_71\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 34.954394 48.160338 \nL 44.698573 93.128243 \nL 54.442752 102.486279 \nL 64.186931 105.517046 \nL 73.93111 111.715817 \nL 83.675289 111.435921 \nL 93.419468 115.799891 \nL 103.163647 117.005553 \nL 112.907826 121.163234 \nL 122.652006 119.718018 \nL 132.396185 125.335394 \nL 142.140364 123.985726 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_72\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 95.714911 \nL 69.163125 99.66838 \nL 88.693125 100.613576 \nL 108.223125 106.10921 \nL 127.753125 97.584144 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_73\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 18.878209 \nL 69.163125 18.488269 \nL 88.693125 17.039922 \nL 108.223125 14.714209 \nL 127.753125 17.207039 \n\" style=\"fill:none;stroke:#2ca02c;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_74\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 34.954394 48.160338 \nL 44.698573 93.128243 \nL 54.442752 102.486279 \nL 64.186931 105.517046 \nL 73.93111 111.715817 \nL 83.675289 111.435921 \nL 93.419468 115.799891 \nL 103.163647 117.005553 \nL 112.907826 121.163234 \nL 122.652006 119.718018 \nL 132.396185 125.335394 \nL 142.140364 123.985726 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_75\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 95.714911 \nL 69.163125 99.66838 \nL 88.693125 100.613576 \nL 108.223125 106.10921 \nL 127.753125 97.584144 \nL 147.283125 95.713394 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_76\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 18.878209 \nL 69.163125 18.488269 \nL 88.693125 17.039922 \nL 108.223125 14.714209 \nL 127.753125 17.207039 \n\" style=\"fill:none;stroke:#2ca02c;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_77\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 34.954394 48.160338 \nL 44.698573 93.128243 \nL 54.442752 102.486279 \nL 64.186931 105.517046 \nL 73.93111 111.715817 \nL 83.675289 111.435921 \nL 93.419468 115.799891 \nL 103.163647 117.005553 \nL 112.907826 121.163234 \nL 122.652006 119.718018 \nL 132.396185 125.335394 \nL 142.140364 123.985726 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_78\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 95.714911 \nL 69.163125 99.66838 \nL 88.693125 100.613576 \nL 108.223125 106.10921 \nL 127.753125 97.584144 \nL 147.283125 95.713394 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_79\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 18.878209 \nL 69.163125 18.488269 \nL 88.693125 17.039922 \nL 108.223125 14.714209 \nL 127.753125 17.207039 \nL 147.283125 17.931213 \n\" style=\"fill:none;stroke:#2ca02c;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_80\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 34.954394 48.160338 \nL 44.698573 93.128243 \nL 54.442752 102.486279 \nL 64.186931 105.517046 \nL 73.93111 111.715817 \nL 83.675289 111.435921 \nL 93.419468 115.799891 \nL 103.163647 117.005553 \nL 112.907826 121.163234 \nL 122.652006 119.718018 \nL 132.396185 125.335394 \nL 142.140364 123.985726 \nL 151.884543 129.360905 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_81\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 95.714911 \nL 69.163125 99.66838 \nL 88.693125 100.613576 \nL 108.223125 106.10921 \nL 127.753125 97.584144 \nL 147.283125 95.713394 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_82\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 18.878209 \nL 69.163125 18.488269 \nL 88.693125 17.039922 \nL 108.223125 14.714209 \nL 127.753125 17.207039 \nL 147.283125 17.931213 \n\" style=\"fill:none;stroke:#2ca02c;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_83\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 34.954394 48.160338 \nL 44.698573 93.128243 \nL 54.442752 102.486279 \nL 64.186931 105.517046 \nL 73.93111 111.715817 \nL 83.675289 111.435921 \nL 93.419468 115.799891 \nL 103.163647 117.005553 \nL 112.907826 121.163234 \nL 122.652006 119.718018 \nL 132.396185 125.335394 \nL 142.140364 123.985726 \nL 151.884543 129.360905 \nL 161.628722 126.75487 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_84\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 95.714911 \nL 69.163125 99.66838 \nL 88.693125 100.613576 \nL 108.223125 106.10921 \nL 127.753125 97.584144 \nL 147.283125 95.713394 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_85\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 18.878209 \nL 69.163125 18.488269 \nL 88.693125 17.039922 \nL 108.223125 14.714209 \nL 127.753125 17.207039 \nL 147.283125 17.931213 \n\" style=\"fill:none;stroke:#2ca02c;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_86\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 34.954394 48.160338 \nL 44.698573 93.128243 \nL 54.442752 102.486279 \nL 64.186931 105.517046 \nL 73.93111 111.715817 \nL 83.675289 111.435921 \nL 93.419468 115.799891 \nL 103.163647 117.005553 \nL 112.907826 121.163234 \nL 122.652006 119.718018 \nL 132.396185 125.335394 \nL 142.140364 123.985726 \nL 151.884543 129.360905 \nL 161.628722 126.75487 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_87\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 95.714911 \nL 69.163125 99.66838 \nL 88.693125 100.613576 \nL 108.223125 106.10921 \nL 127.753125 97.584144 \nL 147.283125 95.713394 \nL 166.813125 100.637084 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_88\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 18.878209 \nL 69.163125 18.488269 \nL 88.693125 17.039922 \nL 108.223125 14.714209 \nL 127.753125 17.207039 \nL 147.283125 17.931213 \n\" style=\"fill:none;stroke:#2ca02c;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_89\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 34.954394 48.160338 \nL 44.698573 93.128243 \nL 54.442752 102.486279 \nL 64.186931 105.517046 \nL 73.93111 111.715817 \nL 83.675289 111.435921 \nL 93.419468 115.799891 \nL 103.163647 117.005553 \nL 112.907826 121.163234 \nL 122.652006 119.718018 \nL 132.396185 125.335394 \nL 142.140364 123.985726 \nL 151.884543 129.360905 \nL 161.628722 126.75487 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_90\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 95.714911 \nL 69.163125 99.66838 \nL 88.693125 100.613576 \nL 108.223125 106.10921 \nL 127.753125 97.584144 \nL 147.283125 95.713394 \nL 166.813125 100.637084 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_91\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 18.878209 \nL 69.163125 18.488269 \nL 88.693125 17.039922 \nL 108.223125 14.714209 \nL 127.753125 17.207039 \nL 147.283125 17.931213 \nL 166.813125 15.424457 \n\" style=\"fill:none;stroke:#2ca02c;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_92\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 34.954394 48.160338 \nL 44.698573 93.128243 \nL 54.442752 102.486279 \nL 64.186931 105.517046 \nL 73.93111 111.715817 \nL 83.675289 111.435921 \nL 93.419468 115.799891 \nL 103.163647 117.005553 \nL 112.907826 121.163234 \nL 122.652006 119.718018 \nL 132.396185 125.335394 \nL 142.140364 123.985726 \nL 151.884543 129.360905 \nL 161.628722 126.75487 \nL 171.372901 132.441578 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_93\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 95.714911 \nL 69.163125 99.66838 \nL 88.693125 100.613576 \nL 108.223125 106.10921 \nL 127.753125 97.584144 \nL 147.283125 95.713394 \nL 166.813125 100.637084 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_94\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 18.878209 \nL 69.163125 18.488269 \nL 88.693125 17.039922 \nL 108.223125 14.714209 \nL 127.753125 17.207039 \nL 147.283125 17.931213 \nL 166.813125 15.424457 \n\" style=\"fill:none;stroke:#2ca02c;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_95\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 34.954394 48.160338 \nL 44.698573 93.128243 \nL 54.442752 102.486279 \nL 64.186931 105.517046 \nL 73.93111 111.715817 \nL 83.675289 111.435921 \nL 93.419468 115.799891 \nL 103.163647 117.005553 \nL 112.907826 121.163234 \nL 122.652006 119.718018 \nL 132.396185 125.335394 \nL 142.140364 123.985726 \nL 151.884543 129.360905 \nL 161.628722 126.75487 \nL 171.372901 132.441578 \nL 181.11708 130.63421 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_96\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 95.714911 \nL 69.163125 99.66838 \nL 88.693125 100.613576 \nL 108.223125 106.10921 \nL 127.753125 97.584144 \nL 147.283125 95.713394 \nL 166.813125 100.637084 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_97\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 18.878209 \nL 69.163125 18.488269 \nL 88.693125 17.039922 \nL 108.223125 14.714209 \nL 127.753125 17.207039 \nL 147.283125 17.931213 \nL 166.813125 15.424457 \n\" style=\"fill:none;stroke:#2ca02c;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_98\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 34.954394 48.160338 \nL 44.698573 93.128243 \nL 54.442752 102.486279 \nL 64.186931 105.517046 \nL 73.93111 111.715817 \nL 83.675289 111.435921 \nL 93.419468 115.799891 \nL 103.163647 117.005553 \nL 112.907826 121.163234 \nL 122.652006 119.718018 \nL 132.396185 125.335394 \nL 142.140364 123.985726 \nL 151.884543 129.360905 \nL 161.628722 126.75487 \nL 171.372901 132.441578 \nL 181.11708 130.63421 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_99\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 95.714911 \nL 69.163125 99.66838 \nL 88.693125 100.613576 \nL 108.223125 106.10921 \nL 127.753125 97.584144 \nL 147.283125 95.713394 \nL 166.813125 100.637084 \nL 186.343125 104.628202 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_100\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 18.878209 \nL 69.163125 18.488269 \nL 88.693125 17.039922 \nL 108.223125 14.714209 \nL 127.753125 17.207039 \nL 147.283125 17.931213 \nL 166.813125 15.424457 \n\" style=\"fill:none;stroke:#2ca02c;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_101\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 34.954394 48.160338 \nL 44.698573 93.128243 \nL 54.442752 102.486279 \nL 64.186931 105.517046 \nL 73.93111 111.715817 \nL 83.675289 111.435921 \nL 93.419468 115.799891 \nL 103.163647 117.005553 \nL 112.907826 121.163234 \nL 122.652006 119.718018 \nL 132.396185 125.335394 \nL 142.140364 123.985726 \nL 151.884543 129.360905 \nL 161.628722 126.75487 \nL 171.372901 132.441578 \nL 181.11708 130.63421 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_102\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 95.714911 \nL 69.163125 99.66838 \nL 88.693125 100.613576 \nL 108.223125 106.10921 \nL 127.753125 97.584144 \nL 147.283125 95.713394 \nL 166.813125 100.637084 \nL 186.343125 104.628202 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_103\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 18.878209 \nL 69.163125 18.488269 \nL 88.693125 17.039922 \nL 108.223125 14.714209 \nL 127.753125 17.207039 \nL 147.283125 17.931213 \nL 166.813125 15.424457 \nL 186.343125 14.630651 \n\" style=\"fill:none;stroke:#2ca02c;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_104\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 34.954394 48.160338 \nL 44.698573 93.128243 \nL 54.442752 102.486279 \nL 64.186931 105.517046 \nL 73.93111 111.715817 \nL 83.675289 111.435921 \nL 93.419468 115.799891 \nL 103.163647 117.005553 \nL 112.907826 121.163234 \nL 122.652006 119.718018 \nL 132.396185 125.335394 \nL 142.140364 123.985726 \nL 151.884543 129.360905 \nL 161.628722 126.75487 \nL 171.372901 132.441578 \nL 181.11708 130.63421 \nL 190.861259 135.56469 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_105\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 95.714911 \nL 69.163125 99.66838 \nL 88.693125 100.613576 \nL 108.223125 106.10921 \nL 127.753125 97.584144 \nL 147.283125 95.713394 \nL 166.813125 100.637084 \nL 186.343125 104.628202 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_106\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 18.878209 \nL 69.163125 18.488269 \nL 88.693125 17.039922 \nL 108.223125 14.714209 \nL 127.753125 17.207039 \nL 147.283125 17.931213 \nL 166.813125 15.424457 \nL 186.343125 14.630651 \n\" style=\"fill:none;stroke:#2ca02c;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_107\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 34.954394 48.160338 \nL 44.698573 93.128243 \nL 54.442752 102.486279 \nL 64.186931 105.517046 \nL 73.93111 111.715817 \nL 83.675289 111.435921 \nL 93.419468 115.799891 \nL 103.163647 117.005553 \nL 112.907826 121.163234 \nL 122.652006 119.718018 \nL 132.396185 125.335394 \nL 142.140364 123.985726 \nL 151.884543 129.360905 \nL 161.628722 126.75487 \nL 171.372901 132.441578 \nL 181.11708 130.63421 \nL 190.861259 135.56469 \nL 200.605438 132.846387 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_108\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 95.714911 \nL 69.163125 99.66838 \nL 88.693125 100.613576 \nL 108.223125 106.10921 \nL 127.753125 97.584144 \nL 147.283125 95.713394 \nL 166.813125 100.637084 \nL 186.343125 104.628202 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_109\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 18.878209 \nL 69.163125 18.488269 \nL 88.693125 17.039922 \nL 108.223125 14.714209 \nL 127.753125 17.207039 \nL 147.283125 17.931213 \nL 166.813125 15.424457 \nL 186.343125 14.630651 \n\" style=\"fill:none;stroke:#2ca02c;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_110\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 34.954394 48.160338 \nL 44.698573 93.128243 \nL 54.442752 102.486279 \nL 64.186931 105.517046 \nL 73.93111 111.715817 \nL 83.675289 111.435921 \nL 93.419468 115.799891 \nL 103.163647 117.005553 \nL 112.907826 121.163234 \nL 122.652006 119.718018 \nL 132.396185 125.335394 \nL 142.140364 123.985726 \nL 151.884543 129.360905 \nL 161.628722 126.75487 \nL 171.372901 132.441578 \nL 181.11708 130.63421 \nL 190.861259 135.56469 \nL 200.605438 132.846387 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_111\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 95.714911 \nL 69.163125 99.66838 \nL 88.693125 100.613576 \nL 108.223125 106.10921 \nL 127.753125 97.584144 \nL 147.283125 95.713394 \nL 166.813125 100.637084 \nL 186.343125 104.628202 \nL 205.873125 104.884377 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_112\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 18.878209 \nL 69.163125 18.488269 \nL 88.693125 17.039922 \nL 108.223125 14.714209 \nL 127.753125 17.207039 \nL 147.283125 17.931213 \nL 166.813125 15.424457 \nL 186.343125 14.630651 \n\" style=\"fill:none;stroke:#2ca02c;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_113\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 34.954394 48.160338 \nL 44.698573 93.128243 \nL 54.442752 102.486279 \nL 64.186931 105.517046 \nL 73.93111 111.715817 \nL 83.675289 111.435921 \nL 93.419468 115.799891 \nL 103.163647 117.005553 \nL 112.907826 121.163234 \nL 122.652006 119.718018 \nL 132.396185 125.335394 \nL 142.140364 123.985726 \nL 151.884543 129.360905 \nL 161.628722 126.75487 \nL 171.372901 132.441578 \nL 181.11708 130.63421 \nL 190.861259 135.56469 \nL 200.605438 132.846387 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_114\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 95.714911 \nL 69.163125 99.66838 \nL 88.693125 100.613576 \nL 108.223125 106.10921 \nL 127.753125 97.584144 \nL 147.283125 95.713394 \nL 166.813125 100.637084 \nL 186.343125 104.628202 \nL 205.873125 104.884377 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_115\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 18.878209 \nL 69.163125 18.488269 \nL 88.693125 17.039922 \nL 108.223125 14.714209 \nL 127.753125 17.207039 \nL 147.283125 17.931213 \nL 166.813125 15.424457 \nL 186.343125 14.630651 \nL 205.873125 13.377273 \n\" style=\"fill:none;stroke:#2ca02c;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_116\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 34.954394 48.160338 \nL 44.698573 93.128243 \nL 54.442752 102.486279 \nL 64.186931 105.517046 \nL 73.93111 111.715817 \nL 83.675289 111.435921 \nL 93.419468 115.799891 \nL 103.163647 117.005553 \nL 112.907826 121.163234 \nL 122.652006 119.718018 \nL 132.396185 125.335394 \nL 142.140364 123.985726 \nL 151.884543 129.360905 \nL 161.628722 126.75487 \nL 171.372901 132.441578 \nL 181.11708 130.63421 \nL 190.861259 135.56469 \nL 200.605438 132.846387 \nL 210.349618 136.922727 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_117\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 95.714911 \nL 69.163125 99.66838 \nL 88.693125 100.613576 \nL 108.223125 106.10921 \nL 127.753125 97.584144 \nL 147.283125 95.713394 \nL 166.813125 100.637084 \nL 186.343125 104.628202 \nL 205.873125 104.884377 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_118\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 18.878209 \nL 69.163125 18.488269 \nL 88.693125 17.039922 \nL 108.223125 14.714209 \nL 127.753125 17.207039 \nL 147.283125 17.931213 \nL 166.813125 15.424457 \nL 186.343125 14.630651 \nL 205.873125 13.377273 \n\" style=\"fill:none;stroke:#2ca02c;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_119\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 34.954394 48.160338 \nL 44.698573 93.128243 \nL 54.442752 102.486279 \nL 64.186931 105.517046 \nL 73.93111 111.715817 \nL 83.675289 111.435921 \nL 93.419468 115.799891 \nL 103.163647 117.005553 \nL 112.907826 121.163234 \nL 122.652006 119.718018 \nL 132.396185 125.335394 \nL 142.140364 123.985726 \nL 151.884543 129.360905 \nL 161.628722 126.75487 \nL 171.372901 132.441578 \nL 181.11708 130.63421 \nL 190.861259 135.56469 \nL 200.605438 132.846387 \nL 210.349618 136.922727 \nL 220.093797 135.125749 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_120\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 95.714911 \nL 69.163125 99.66838 \nL 88.693125 100.613576 \nL 108.223125 106.10921 \nL 127.753125 97.584144 \nL 147.283125 95.713394 \nL 166.813125 100.637084 \nL 186.343125 104.628202 \nL 205.873125 104.884377 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_121\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 18.878209 \nL 69.163125 18.488269 \nL 88.693125 17.039922 \nL 108.223125 14.714209 \nL 127.753125 17.207039 \nL 147.283125 17.931213 \nL 166.813125 15.424457 \nL 186.343125 14.630651 \nL 205.873125 13.377273 \n\" style=\"fill:none;stroke:#2ca02c;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_122\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 34.954394 48.160338 \nL 44.698573 93.128243 \nL 54.442752 102.486279 \nL 64.186931 105.517046 \nL 73.93111 111.715817 \nL 83.675289 111.435921 \nL 93.419468 115.799891 \nL 103.163647 117.005553 \nL 112.907826 121.163234 \nL 122.652006 119.718018 \nL 132.396185 125.335394 \nL 142.140364 123.985726 \nL 151.884543 129.360905 \nL 161.628722 126.75487 \nL 171.372901 132.441578 \nL 181.11708 130.63421 \nL 190.861259 135.56469 \nL 200.605438 132.846387 \nL 210.349618 136.922727 \nL 220.093797 135.125749 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_123\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 95.714911 \nL 69.163125 99.66838 \nL 88.693125 100.613576 \nL 108.223125 106.10921 \nL 127.753125 97.584144 \nL 147.283125 95.713394 \nL 166.813125 100.637084 \nL 186.343125 104.628202 \nL 205.873125 104.884377 \nL 225.403125 99.310135 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_124\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 18.878209 \nL 69.163125 18.488269 \nL 88.693125 17.039922 \nL 108.223125 14.714209 \nL 127.753125 17.207039 \nL 147.283125 17.931213 \nL 166.813125 15.424457 \nL 186.343125 14.630651 \nL 205.873125 13.377273 \n\" style=\"fill:none;stroke:#2ca02c;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_125\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 34.954394 48.160338 \nL 44.698573 93.128243 \nL 54.442752 102.486279 \nL 64.186931 105.517046 \nL 73.93111 111.715817 \nL 83.675289 111.435921 \nL 93.419468 115.799891 \nL 103.163647 117.005553 \nL 112.907826 121.163234 \nL 122.652006 119.718018 \nL 132.396185 125.335394 \nL 142.140364 123.985726 \nL 151.884543 129.360905 \nL 161.628722 126.75487 \nL 171.372901 132.441578 \nL 181.11708 130.63421 \nL 190.861259 135.56469 \nL 200.605438 132.846387 \nL 210.349618 136.922727 \nL 220.093797 135.125749 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_126\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 95.714911 \nL 69.163125 99.66838 \nL 88.693125 100.613576 \nL 108.223125 106.10921 \nL 127.753125 97.584144 \nL 147.283125 95.713394 \nL 166.813125 100.637084 \nL 186.343125 104.628202 \nL 205.873125 104.884377 \nL 225.403125 99.310135 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_127\">\n    <path clip-path=\"url(#pc1c6058863)\" d=\"M 49.633125 18.878209 \nL 69.163125 18.488269 \nL 88.693125 17.039922 \nL 108.223125 14.714209 \nL 127.753125 17.207039 \nL 147.283125 17.931213 \nL 166.813125 15.424457 \nL 186.343125 14.630651 \nL 205.873125 13.377273 \nL 225.403125 15.41053 \n\" style=\"fill:none;stroke:#2ca02c;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"patch_3\">\n    <path d=\"M 30.103125 143.1 \nL 30.103125 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_4\">\n    <path d=\"M 225.403125 143.1 \nL 225.403125 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_5\">\n    <path d=\"M 30.103125 143.1 \nL 225.403125 143.1 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_6\">\n    <path d=\"M 30.103125 7.2 \nL 225.403125 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"legend_1\">\n    <g id=\"patch_7\">\n     <path d=\"M 138.8125 99.084375 \nL 218.403125 99.084375 \nQ 220.403125 99.084375 220.403125 97.084375 \nL 220.403125 53.215625 \nQ 220.403125 51.215625 218.403125 51.215625 \nL 138.8125 51.215625 \nQ 136.8125 51.215625 136.8125 53.215625 \nL 136.8125 97.084375 \nQ 136.8125 99.084375 138.8125 99.084375 \nz\n\" style=\"fill:#ffffff;opacity:0.8;stroke:#cccccc;stroke-linejoin:miter;\"/>\n    </g>\n    <g id=\"line2d_128\">\n     <path d=\"M 140.8125 59.314062 \nL 160.8125 59.314062 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n    </g>\n    <g id=\"line2d_129\"/>\n    <g id=\"text_13\">\n     <!-- train_loss -->\n     <g transform=\"translate(168.8125 62.814062)scale(0.1 -0.1)\">\n      <defs>\n       <path d=\"M 18.3125 70.21875 \nL 18.3125 54.6875 \nL 36.8125 54.6875 \nL 36.8125 47.703125 \nL 18.3125 47.703125 \nL 18.3125 18.015625 \nQ 18.3125 11.328125 20.140625 9.421875 \nQ 21.96875 7.515625 27.59375 7.515625 \nL 36.8125 7.515625 \nL 36.8125 0 \nL 27.59375 0 \nQ 17.1875 0 13.234375 3.875 \nQ 9.28125 7.765625 9.28125 18.015625 \nL 9.28125 47.703125 \nL 2.6875 47.703125 \nL 2.6875 54.6875 \nL 9.28125 54.6875 \nL 9.28125 70.21875 \nz\n\" id=\"DejaVuSans-116\"/>\n       <path d=\"M 41.109375 46.296875 \nQ 39.59375 47.171875 37.8125 47.578125 \nQ 36.03125 48 33.890625 48 \nQ 26.265625 48 22.1875 43.046875 \nQ 18.109375 38.09375 18.109375 28.8125 \nL 18.109375 0 \nL 9.078125 0 \nL 9.078125 54.6875 \nL 18.109375 54.6875 \nL 18.109375 46.1875 \nQ 20.953125 51.171875 25.484375 53.578125 \nQ 30.03125 56 36.53125 56 \nQ 37.453125 56 38.578125 55.875 \nQ 39.703125 55.765625 41.0625 55.515625 \nz\n\" id=\"DejaVuSans-114\"/>\n       <path d=\"M 34.28125 27.484375 \nQ 23.390625 27.484375 19.1875 25 \nQ 14.984375 22.515625 14.984375 16.5 \nQ 14.984375 11.71875 18.140625 8.90625 \nQ 21.296875 6.109375 26.703125 6.109375 \nQ 34.1875 6.109375 38.703125 11.40625 \nQ 43.21875 16.703125 43.21875 25.484375 \nL 43.21875 27.484375 \nz\nM 52.203125 31.203125 \nL 52.203125 0 \nL 43.21875 0 \nL 43.21875 8.296875 \nQ 40.140625 3.328125 35.546875 0.953125 \nQ 30.953125 -1.421875 24.3125 -1.421875 \nQ 15.921875 -1.421875 10.953125 3.296875 \nQ 6 8.015625 6 15.921875 \nQ 6 25.140625 12.171875 29.828125 \nQ 18.359375 34.515625 30.609375 34.515625 \nL 43.21875 34.515625 \nL 43.21875 35.40625 \nQ 43.21875 41.609375 39.140625 45 \nQ 35.0625 48.390625 27.6875 48.390625 \nQ 23 48.390625 18.546875 47.265625 \nQ 14.109375 46.140625 10.015625 43.890625 \nL 10.015625 52.203125 \nQ 14.9375 54.109375 19.578125 55.046875 \nQ 24.21875 56 28.609375 56 \nQ 40.484375 56 46.34375 49.84375 \nQ 52.203125 43.703125 52.203125 31.203125 \nz\n\" id=\"DejaVuSans-97\"/>\n       <path d=\"M 9.421875 54.6875 \nL 18.40625 54.6875 \nL 18.40625 0 \nL 9.421875 0 \nz\nM 9.421875 75.984375 \nL 18.40625 75.984375 \nL 18.40625 64.59375 \nL 9.421875 64.59375 \nz\n\" id=\"DejaVuSans-105\"/>\n       <path d=\"M 54.890625 33.015625 \nL 54.890625 0 \nL 45.90625 0 \nL 45.90625 32.71875 \nQ 45.90625 40.484375 42.875 44.328125 \nQ 39.84375 48.1875 33.796875 48.1875 \nQ 26.515625 48.1875 22.3125 43.546875 \nQ 18.109375 38.921875 18.109375 30.90625 \nL 18.109375 0 \nL 9.078125 0 \nL 9.078125 54.6875 \nL 18.109375 54.6875 \nL 18.109375 46.1875 \nQ 21.34375 51.125 25.703125 53.5625 \nQ 30.078125 56 35.796875 56 \nQ 45.21875 56 50.046875 50.171875 \nQ 54.890625 44.34375 54.890625 33.015625 \nz\n\" id=\"DejaVuSans-110\"/>\n       <path d=\"M 50.984375 -16.609375 \nL 50.984375 -23.578125 \nL -0.984375 -23.578125 \nL -0.984375 -16.609375 \nz\n\" id=\"DejaVuSans-95\"/>\n       <path d=\"M 9.421875 75.984375 \nL 18.40625 75.984375 \nL 18.40625 0 \nL 9.421875 0 \nz\n\" id=\"DejaVuSans-108\"/>\n       <path d=\"M 44.28125 53.078125 \nL 44.28125 44.578125 \nQ 40.484375 46.53125 36.375 47.5 \nQ 32.28125 48.484375 27.875 48.484375 \nQ 21.1875 48.484375 17.84375 46.4375 \nQ 14.5 44.390625 14.5 40.28125 \nQ 14.5 37.15625 16.890625 35.375 \nQ 19.28125 33.59375 26.515625 31.984375 \nL 29.59375 31.296875 \nQ 39.15625 29.25 43.1875 25.515625 \nQ 47.21875 21.78125 47.21875 15.09375 \nQ 47.21875 7.46875 41.1875 3.015625 \nQ 35.15625 -1.421875 24.609375 -1.421875 \nQ 20.21875 -1.421875 15.453125 -0.5625 \nQ 10.6875 0.296875 5.421875 2 \nL 5.421875 11.28125 \nQ 10.40625 8.6875 15.234375 7.390625 \nQ 20.0625 6.109375 24.8125 6.109375 \nQ 31.15625 6.109375 34.5625 8.28125 \nQ 37.984375 10.453125 37.984375 14.40625 \nQ 37.984375 18.0625 35.515625 20.015625 \nQ 33.0625 21.96875 24.703125 23.78125 \nL 21.578125 24.515625 \nQ 13.234375 26.265625 9.515625 29.90625 \nQ 5.8125 33.546875 5.8125 39.890625 \nQ 5.8125 47.609375 11.28125 51.796875 \nQ 16.75 56 26.8125 56 \nQ 31.78125 56 36.171875 55.265625 \nQ 40.578125 54.546875 44.28125 53.078125 \nz\n\" id=\"DejaVuSans-115\"/>\n      </defs>\n      <use xlink:href=\"#DejaVuSans-116\"/>\n      <use x=\"39.208984\" xlink:href=\"#DejaVuSans-114\"/>\n      <use x=\"80.322266\" xlink:href=\"#DejaVuSans-97\"/>\n      <use x=\"141.601562\" xlink:href=\"#DejaVuSans-105\"/>\n      <use x=\"169.384766\" xlink:href=\"#DejaVuSans-110\"/>\n      <use x=\"232.763672\" xlink:href=\"#DejaVuSans-95\"/>\n      <use x=\"282.763672\" xlink:href=\"#DejaVuSans-108\"/>\n      <use x=\"310.546875\" xlink:href=\"#DejaVuSans-111\"/>\n      <use x=\"371.728516\" xlink:href=\"#DejaVuSans-115\"/>\n      <use x=\"423.828125\" xlink:href=\"#DejaVuSans-115\"/>\n     </g>\n    </g>\n    <g id=\"line2d_130\">\n     <path d=\"M 140.8125 74.270312 \nL 160.8125 74.270312 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n    </g>\n    <g id=\"line2d_131\"/>\n    <g id=\"text_14\">\n     <!-- val_loss -->\n     <g transform=\"translate(168.8125 77.770312)scale(0.1 -0.1)\">\n      <defs>\n       <path d=\"M 2.984375 54.6875 \nL 12.5 54.6875 \nL 29.59375 8.796875 \nL 46.6875 54.6875 \nL 56.203125 54.6875 \nL 35.6875 0 \nL 23.484375 0 \nz\n\" id=\"DejaVuSans-118\"/>\n      </defs>\n      <use xlink:href=\"#DejaVuSans-118\"/>\n      <use x=\"59.179688\" xlink:href=\"#DejaVuSans-97\"/>\n      <use x=\"120.458984\" xlink:href=\"#DejaVuSans-108\"/>\n      <use x=\"148.242188\" xlink:href=\"#DejaVuSans-95\"/>\n      <use x=\"198.242188\" xlink:href=\"#DejaVuSans-108\"/>\n      <use x=\"226.025391\" xlink:href=\"#DejaVuSans-111\"/>\n      <use x=\"287.207031\" xlink:href=\"#DejaVuSans-115\"/>\n      <use x=\"339.306641\" xlink:href=\"#DejaVuSans-115\"/>\n     </g>\n    </g>\n    <g id=\"line2d_132\">\n     <path d=\"M 140.8125 89.226562 \nL 160.8125 89.226562 \n\" style=\"fill:none;stroke:#2ca02c;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n    </g>\n    <g id=\"line2d_133\"/>\n    <g id=\"text_15\">\n     <!-- val_acc -->\n     <g transform=\"translate(168.8125 92.726562)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-118\"/>\n      <use x=\"59.179688\" xlink:href=\"#DejaVuSans-97\"/>\n      <use x=\"120.458984\" xlink:href=\"#DejaVuSans-108\"/>\n      <use x=\"148.242188\" xlink:href=\"#DejaVuSans-95\"/>\n      <use x=\"198.242188\" xlink:href=\"#DejaVuSans-97\"/>\n      <use x=\"259.521484\" xlink:href=\"#DejaVuSans-99\"/>\n      <use x=\"314.501953\" xlink:href=\"#DejaVuSans-99\"/>\n     </g>\n    </g>\n   </g>\n  </g>\n </g>\n <defs>\n  <clipPath id=\"pc1c6058863\">\n   <rect height=\"135.9\" width=\"195.3\" x=\"30.103125\" y=\"7.2\"/>\n  </clipPath>\n </defs>\n</svg>\n",
   "text/plain": "<Figure size 252x180 with 1 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 }
]
```

## Summary




## Exercises



:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/)
:end_tab:
