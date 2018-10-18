# Networks Using Duplicates (VGG)

AlexNet adds three convolutional layers to LeNet. However, the author of AlexNet made significant adjustments to the convolution windows, number of output channels, and construction order of the layers. Although AlexNet proved that deep convolutional neural networks can achieve good results, it does not provide simple rules to guide subsequent researchers in the design of new networks. In the following sections, we will introduce several different concepts used in deep network design.

In this section, we will introduce VGG, named after the Visual Geometry Group[1], the laboratory where the author of this thesis works. VGG proposed the idea of constructing a deep model by repeatedly using simple foundation blocks.

## VGG Blocks

VGG block construction rules: After using several identical convolutional layers in succession with paddings of 1 and window shapes of $3\times 3$, attach a maximum pooling layer with a stride of 2 and a window shape of $2\times 2$. For the convolutional layers, keep the entered values of height and width unchanged, but halve the entered height and width for the pooling layer. We use the `vgg_block` function to implement this basic VGG block. This function can specify the number of convolutional layers `num_convs` and the number of output channels `num_channels`.

```{.python .input  n=1}
import gluonbook as gb
from mxnet import gluon, init, nd
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

Like AlexNet and LeNet, the VGG Network is composed of convolutional layer modules attached to fully connected layer modules. Several `vgg_block` are connected in series in the convolutional layer module, the hyper-parameter of which is defined by the variable `conv_arch`. This variable specifies the numbers of convolutional layers and output channels in each VGG block. The fully connected module is the same as that of AlexNet.

Now, we will create a VGG network. This network has 5 convolutional blocks, among which the former two use a single convolutional layer, while the latter three use a double convolutional layer. The first block has 64 output channels, and the latter blocks double the number of output channels, until that number reaches 512. Since this network uses 8 convolutional layers and 3 fully connected layers, it is often called VGG-11.

```{.python .input  n=2}
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
```

Now, we will implement VGG-11.

```{.python .input  n=3}
def vgg(conv_arch):
    net = nn.Sequential()
    # The convolutional layer part.
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))
    # The fully connected layer part.
    net.add(nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(10))
    return net

net = vgg(conv_arch)
```

Next, we will construct a single-channel data example with a height and width of 224 to observe the output shape of each layer.

```{.python .input  n=4}
net.initialize()
X = nd.random.uniform(shape=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.name, 'output shape:\t', X.shape)
```

As we can see, we halve the entered value of the height and width each time, until the final values of height and width change to 7 before we pass it to the fully connected layer. Meanwhile, the number of output channels doubles until it becomes 512. Since the windows of each convolutional layer are of the same size, the model parameter size of each layer and the computational complexity is proportional to the product of height, width, number of input channels, and number of output channels. By halving the height and width while doubling the number of channels, VGG allows most convolutional layers to have the same model parameter size and computational complexity.

## Model Training

Since VGG-11 is more complicated than AlexNet in terms of computation, for testing purposes, we construct a network with a smaller number of channels. In other words, we use a narrower network to train the model using the Fashion-MNIST data set.

```{.python .input  n=5}
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)
```

Apart from using a slightly larger learning rate, the model training process is similar to that of AlexNet in the last section.

```{.python .input}
lr, num_epochs, batch_size, ctx = 0.05, 5, 128, gb.try_gpu()
net.initialize(ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size, resize=224)
gb.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)
```

## Summary

* VGG-11 constructs a network using 5 reusable convolutional blocks. Different VGG models can be defined by the differences in number of convolutional layers and output channels in each block.

## exercise

* Compared with AlexNet, VGG is much slower in terms of computation, and it also needs more GPU memory. Try to analyze the reasons for this.
* Try to change the height and width of the images in Fashion-MNIST from 224 to 96. What influence does this have on the experiments?
* Refer to Table 1 in the VGG thesis to construct other common models, e.g. VGG-16 and VGG-19[1].

## Scan the QR Code to Access [Discussions](https://discuss.gluon.ai/t/topic/1277)

![](../img/qr_vgg.svg)

## References

[1] Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.
