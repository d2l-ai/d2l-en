# Designing Convolution Network Architectures
:label:`sec_design-space`



## Bottleneck Residual Blocks with Group Convolution

```{.python .input  n=1}
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
        Y += X
        return F.relu(Y)
```

Now let's look at a situation where the input and output are of the same shape.

```{.python .input  n=2}
blk = GroupConvBotResidual(32, 32, 16, 1)
X = d2l.randn(4, 32, 96, 96)
blk(X).shape
```

We also have the option to halve the output height and width while increasing the number of output channels.

```{.python .input  n=3}
blk = GroupConvBotResidual(32, 32, 16, 1, use_1x1conv=True, strides=2)
blk(X).shape
```

## RegNet

Now, we implement this module. Note that special processing has been performed on the first module.

```{.python .input  n=4}
class AnyNet(d2l.Classifier):
    def stem(self, input_channels, num_channels):
        return nn.Sequential(
            nn.Conv2d(input_channels, num_channels, kernel_size=3, stride=2,
                      padding=1),
            nn.BatchNorm2d(num_channels), nn.ReLU())
```

other block

```{.python .input  n=5}
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
@d2l.add_to_class(AnyNet)
def __init__(self, arch, stem_channels, num_classes=10, lr=0.1):
    super(AnyNet, self).__init__()
    self.save_hyperparameters()
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
class RegNet32(AnyNet):
    def __init__(self, num_classes=10, lr=0.1):
        stem_channels, group_width, bot_mul = 32, 16, 1
        depths, channels = [4, 6], [32, 80]
        super().__init__(
            ((depths[0], stem_channels, channels[0], group_width, bot_mul),
             (depths[1], channels[0], channels[1], group_width, bot_mul)),
            stem_channels, num_classes, lr)
```

```{.python .input  n=8}
RegNet32().layer_summary((1, 1, 96, 96))
```

## Training

We train RegNet on the Fashion-MNIST dataset, just like before.

```{.python .input  n=9}
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
