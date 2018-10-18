# Residual Networks (ResNet)

Let us start with a question: Can we add a new layer to the neural network so that the fully trained model can reduce training errors more effectively? In theory, the space of the original model solution is only the subspace of the space of the new model solution. This means that if we can train the newly-added layer into an identity mapping $f(x) = x$, the new model will be as effective as the original model. As the new model may get a better solution to fit the training data set, the added layer might make it easier to reduce training errors. In practice, however, with the addition of too many layers, training errors increase rather than decrease. Even if the numerical stability brought about by batch normalization makes it easier to train a deep model, this problem still exists. In response to this problem, He Kaiming and his colleagues proposed the ResNet[1]. It won the ImageNet Visual Recognition Challenge in 2015 and had a profound influence on the design of subsequent deep neural networks.


## Residual Blocks

Let us focus on the local neural network. As shown in Figure 5.9, set the input as $\boldsymbol{x}$. We assume the ideal mapping we want to obtain by learning is $f(\boldsymbol{x})$, to be used as the input to the activation function in Figure 5.9 above. The portion within the dotted-line box in the left image must directly fit the mapping $f(\boldsymbol{x})$. The portion within the dotted-line box in the right image must fit the residual mapping $f(\boldsymbol{x})-\boldsymbol{x}$. In practice, the residual mapping is often easier to optimize. Use the identity mapping mentioned at the beginning of this section as the ideal mapping $f(\boldsymbol{x})$ that we want to obtain by learning, and use ReLU as the activation function. We only need to zero the weight and the bias parameter of the weighting operation (such as affine) at the top of the right image in Figure 5.9, so the output of ReLU above is identical to the input $\boldsymbol{x}$. The right image in Figure 5.9 is also the basic block of ResNet, that is, the residual block. In the residual block, the input can travel forward faster through cross-layer data paths.

![Set the input as $\boldsymbol{x}$. We assumer that the ideal mapping of ReLU at the top of the figure is $f(\boldsymbol{x})$. The portion within the dotted-line box in the left image must directly fit the mapping $f(\boldsymbol{x})$. The portion within the dotted-line box in the right image must fit the residual mapping $f(\boldsymbol{x})-\boldsymbol{x}$. ](../img/residual-block.svg)

ResNet follows VGG's full $3\times 3$ convolutional layer design. The residual block has two $3\times 3$ convolutional layers with the same number of output channels. Each convolutional layer is followed by a batch normalization layer and a ReLU activation function. Then, we skip these two convolution operations and add the input directly before the final ReLU activation function. This kind of design requires that the output of the two convolutional layers be of the same shape as the input, so that they can be added together. If you want to change the number of channels, you need to introduce an additional $1\times 1$ convolutional layer to transform the input into the desired shape for the addition operation.

The residual block is implemented as follows. It can be used to set the number of output channels, whether to use an additional $1\times 1$ convolutional layer to change the number of channels, as well as the stride of convolutional layers.

```{.python .input  n=1}
import gluonbook as gb
from mxnet import gluon, init, nd
from mxnet.gluon import nn

class Residual(nn.Block):  # This category has been saved in the gluonbook package for future use.
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1,
                               strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1,
                                   strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def forward(self, X):
        Y = nd.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return nd.relu(Y + X)
```

Now let us look at a situation where the input and output are of the same shape.

```{.python .input  n=2}
blk = Residual(3)
blk.initialize()
X = nd.random.uniform(shape=(4, 3, 6, 6))
blk(X).shape
```

We also have the option to halve the output height and width while increasing the number of output channels.

```{.python .input  n=3}
blk = Residual(6, use_1x1conv=True, strides=2)
blk.initialize()
blk(X).shape
```

## ResNet Model

The first two layers of ResNet are the same as those of the GoogLeNet we described before: the $7\times 7$ convolutional layer with 64 output channels and a stride of 2 is followed by the $3\times 3$ maximum pooling layer with a stride of 2. The difference is the batch normalization layer added after each convolutional layer in ResNet.

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
        nn.BatchNorm(), nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

GoogLeNet uses four blocks made up of Inception blocks. However, ResNet uses four modules made up of residual blocks, each of which uses several residual blocks with the same number of output channels. The number of channels in the first module is the same as the number of input channels. Since a maximum pooling layer with a stride of 2 has already been used, it is not necessary to reduce the height and width. In the first residual block for each of the subsequent modules, the number of channels is doubled compared with that of the previous module, and the height and width are halved.

Now, we implement this module. Note that special processing has been performed on the first module.

```{.python .input  n=4}
def resnet_block(num_channels, num_residuals, first_block=False):
    blk = nn.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.add(Residual(num_channels))
    return blk
```

Then, we add all the residual blocks to ResNet. Here, two residual blocks are used for each module.

```{.python .input  n=5}
net.add(resnet_block(64, 2, first_block=True),
        resnet_block(128, 2),
        resnet_block(256, 2),
        resnet_block(512, 2))
```

Finally, just like GoogLeNet, we add a global average pooling layer, followed by the fully connected layer output.

```{.python .input}
net.add(nn.GlobalAvgPool2D(), nn.Dense(10))
```

There are 4 convolutional layers in each module (excluding the $1\times 1$ convolutional layer). Together with the first convolutional layer and the final fully connected layer, there are 18 layers in total. Therefore, this model is commonly known as ResNet-18. By configuring different numbers of channels and residual blocks in the module, we can create different ResNet models, such as the deeper 152-layer ResNet-152. Although the main architecture of ResNet is similar to that of GoogLeNet, ResNet's structure is simpler and easier to modify. All these factors have resulted in the rapid and widespread use of ResNet.

Before training ResNet, let us observe how the input shape changes between different modules in ResNet.

```{.python .input  n=6}
X = nd.random.uniform(shape=(1, 1, 224, 224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

## Data Acquisition and Training

Now we train ResNet on the Fashion-MNIST data set.

```{.python .input}
lr, num_epochs, batch_size, ctx = 0.05, 5, 256, gb.try_gpu()
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size, resize=96)
gb.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)
```

## Summary

* We can train an effective deep neural network by having residual blocks pass through cross-layer data channels.
* ResNet has had a major influence on the design of subsequent deep neural networks.


## exercise

* Refer to Table 1 in the ResNet thesis to implement different versions of ResNet[1].
* For deeper networks, the ResNet thesis introduces a "bottleneck" architecture to reduce model complexity. Try to implement it [1].
* In subsequent versions of ResNet, the author changed the "convolution, batch normalization, and activation" architecture to the "batch normalization, activation, and convolution" architecture. Make this improvement yourself([2], Figure 1).

## Scan the QR Code to Access [Discussions](https://discuss.gluon.ai/t/topic/1663)

![](../img/qr_resnet.svg)

## References

[1] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).

[2] He, K., Zhang, X., Ren, S., & Sun, J. (2016, October). Identity mappings in deep residual networks. In European Conference on Computer Vision (pp. 630-645). Springer, Cham.
