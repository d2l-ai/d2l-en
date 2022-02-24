```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch'])
```

# Designing Convolution Network Architectures
:label:`sec_design-space`



## Bottleneck Residual Blocks with Group Convolution

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx, init
from mxnet.gluon import nn
npx.set_np()

class GroupConvBotResidual(nn.Block):
    """The bottleneck residual block with group convolution."""
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

```{.python .input  n=1}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F

class GroupConvBotResidual(nn.Module):
    """The bottleneck residual block with group convolution."""
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

Now let's look at a situation where the input and output are of the same shape.

```{.python .input  n=2}
%%tab all
if tab.selected('mxnet'):
    blk = GroupConvBotResidual(32, 16, 1)
    blk.initialize()
if tab.selected('pytorch'):
    blk = GroupConvBotResidual(32, 32, 16, 1)
X = d2l.randn(4, 32, 96, 96)
blk(X).shape
```

We also have the option to halve the output height and width while increasing the number of output channels.

```{.python .input  n=3}
%%tab all
if tab.selected('mxnet'):
    blk = GroupConvBotResidual(32, 16, 1, use_1x1conv=True, strides=2)
    blk.initialize()
if tab.selected('pytorch'):
    blk = GroupConvBotResidual(32, 32, 16, 1, use_1x1conv=True, strides=2)
blk(X).shape
```

## RegNet

Now, we implement this module. Note that special processing has been performed on the first module.

```{.python .input  n=4}
%%tab mxnet
class AnyNet(d2l.Classifier):
    def stem(self, num_channels):
        net = nn.Sequential()
        net.add(nn.Conv2D(num_channels, kernel_size=3, padding=1, strides=2),
                nn.BatchNorm(), nn.Activation('relu'))
        return net
```

```{.python .input}
%%tab pytorch
class AnyNet(d2l.Classifier):
    def stem(self, input_channels, num_channels):
        return nn.Sequential(
            nn.Conv2d(input_channels, num_channels, kernel_size=3, stride=2,
                      padding=1),
            nn.BatchNorm2d(num_channels), nn.ReLU())
```

other block

```{.python .input}
%%tab mxnet
@d2l.add_to_class(AnyNet)
def stage(self, depth, num_channels, group_width, bot_mul):
    net = nn.Sequential()
    for i in range(depth):
        if i == 0:
            net.add(GroupConvBotResidual(
                num_channels, group_width, bot_mul, use_1x1conv=True,
                strides=2))
        else:
            net.add(GroupConvBotResidual(
                num_channels, num_channels, group_width, bot_mul))
    return net
```

```{.python .input  n=5}
%%tab pytorch
@d2l.add_to_class(AnyNet)
def stage(self, depth, input_channels, num_channels, group_width, bot_mul):
    blk = []
    for i in range(depth):
        if i == 0:
            blk.append(GroupConvBotResidual(
                input_channels, num_channels, group_width, bot_mul,
                use_1x1conv=True, strides=2))
        else:
            blk.append(GroupConvBotResidual(
                num_channels, num_channels, group_width, bot_mul))
    return nn.Sequential(*blk)
```

Then, we add all the modules to RegNet.

```{.python .input  n=6}
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

Before training RegNet, let's observe how the input shape changes across different modules in ResNet.

```{.python .input  n=7}
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

```{.python .input  n=8}
%%tab all
RegNet32().layer_summary((1, 1, 96, 96))
```

## Training

We train RegNet on the Fashion-MNIST dataset, just like before.

```{.python .input  n=9}
%%tab all
model = RegNet32(lr=0.05)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
trainer.fit(model, data)
```

## Summary




## Exercises



:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/)
:end_tab:
