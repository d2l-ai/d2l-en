# Network with Parallel Concatenations (GoogLeNet)

During the ImageNet Large Scale Visual Recognition Challenge 2014, a network architecture called GoogLeNet outshone its peers[1]. Although it pays tribute to LeNet in name, there are hardly any traces of LeNet in the network architecture. GoogLeNet absorbed the idea of serial networks in NiN and used this as the foundation for major improvements. Over the next few years, researchers made several improvements to GoogLeNet. In this section, we will introduce the first version of this model series.


## Inception Blocks

The basic convolutional block in GoogLeNet is called an Inception block, named after the movie of the same name. This basic block is more complex in structure than the NiN block described in the previous section.

![Structure of the Inception block. ](../img/inception.svg)

As can be seen from Figure 5.8, there are four parallel paths in the Inception block. The first three paths use convolutional layers with window sizes of $1\times 1$, $3\times 3$, and $5\times 5$ to extract information from different spatial sizes. The middle two paths will perform a $1\times 1$ convolution on the input to reduce the number of input channels, so as to reduce the model's complexity. The fourth path uses the $3\times 3$ maximum pooling layer, followed by the $1\times 1$ convolutional layer, to change the number of channels. The four paths all use appropriate padding to give the input and output the same height and width. Finally, we concatenate the output of each path on the channel dimension and input it to the next layer.

The customizable hyper-parameters in the Inception block are the number of output channels per layer, which we use to control the model complexity.

```{.python .input  n=1}
import gluonbook as gb
from mxnet import gluon, init, nd
from mxnet.gluon import nn

class Inception(nn.Block):
    # c1 - c4 are the number of output channels for each layer in the path.
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Path 1 is a single 1 x 1 convolutional layer.
        self.p1_1 = nn.Conv2D(c1, kernel_size=1, activation='relu')
        # Path 2 is a 1 x 1 convolutional layer followed by a 3 x 3 convolutional layer.
        self.p2_1 = nn.Conv2D(c2[0], kernel_size=1, activation='relu')
        self.p2_2 = nn.Conv2D(c2[1], kernel_size=3, padding=1,
                              activation='relu')
        # Path 3 is a 1 x 1 convolutional layer followed by a 5 x 5 convolutional layer.
        self.p3_1 = nn.Conv2D(c3[0], kernel_size=1, activation='relu')
        self.p3_2 = nn.Conv2D(c3[1], kernel_size=5, padding=2,
                              activation='relu')
        # Path 4 is a 3 x 3 maximum pooling layer followed by a 1 x 1 convolutional layer.
        self.p4_1 = nn.MaxPool2D(pool_size=3, strides=1, padding=1)
        self.p4_2 = nn.Conv2D(c4, kernel_size=1, activation='relu')

    def forward(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        return nd.concat(p1, p2, p3, p4, dim=1)  # Concatenate the outputs on the channel dimension.
```

## GoogLeNet Model

GoogLeNet, like VGG, uses five blocks in the body convolution section, with a $3\times 3$ maximum pooling layer with a stride of 2 between each block to reduce the output height and width. The first block uses a 64-channel 7×7 convolutional layer.

```{.python .input  n=2}
b1 = nn.Sequential()
b1.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3, activation='relu'),
       nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

The second block uses two convolutional layers: first, a 64-channel $1\times 1$ convolutional layer, then a $3\times 3$ convolutional layer that triples the number of channels. This corresponds to the second path in the Inception block.

```{.python .input  n=3}
b2 = nn.Sequential()
b2.add(nn.Conv2D(64, kernel_size=1),
       nn.Conv2D(192, kernel_size=3, padding=1),
       nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

The third block connects two complete Inception blocks in a series. The number of output channels of the first Inception block is $64+128+32+32=256$, and the ratio to the output channels of the four paths is $64:128:32:32=2:4:1:1$. The second and third paths first reduce the number of input channels to $96/192=1/2$ and $16/192=1/12$, respectively, and then connect the second convolutional layer. The number of output channels of the second Inception block is increased to $128+192+96+64=480$, and the ratio to the number of output channels per path is $128:192:96:64 = 4:6:3:2$. The second and third paths first reduce the number of input channels to $128/256=1/2$ and $32/256=1/8$, respectively.

```{.python .input  n=4}
b3 = nn.Sequential()
b3.add(Inception(64, (96, 128), (16, 32), 32),
       Inception(128, (128, 192), (32, 96), 64),
       nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

The fourth block is more complicated. It connects five Inception blocks in series, and they have $192+208+48+64=512$, $160+224+64+64=512$, $128+256+64+64=512$, $112+288+64+64=528$, and $256+320+128+128=832$ output channels, respectively. The number of channels assigned to these paths is similar to that in the third module: the second path with the $3\times 3$ convolutional layer outputs the largest number of channels, followed by the first path with only the $1\times 1$ convolutional layer, the third path with the $5\times 5$ convolutional layer, and the fourth path with the $3\times 3$ maximum pooling layer. The second and third paths will first reduce the number of channels according the ratio. These ratios are slightly different in different Inception blocks.

```{.python .input  n=5}
b4 = nn.Sequential()
b4.add(Inception(192, (96, 208), (16, 48), 64),
       Inception(160, (112, 224), (24, 64), 64),
       Inception(128, (128, 256), (24, 64), 64),
       Inception(112, (144, 288), (32, 64), 64),
       Inception(256, (160, 320), (32, 128), 128),
       nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

The fifth block has two Inception blocks with $256+320+128+128=832$ and $384+384+128+128=1024$ output channels. The number of channels assigned to each path is the same as that in the third and fourth modules, but differs in specific values. It should be noted that the fifth block is followed by the output layer. This block uses the global average pooling layer to change the height and width of each channel to 1, just as in NiN. Finally, we turn the output into a two-dimensional array followed by a fully connected layer whose number of outputs is the number of label classes.

```{.python .input  n=6}
b5 = nn.Sequential()
b5.add(Inception(256, (160, 320), (32, 128), 128),
       Inception(384, (192, 384), (48, 128), 128),
       nn.GlobalAvgPool2D())

net = nn.Sequential()
net.add(b1, b2, b3, b4, b5, nn.Dense(10))
```

The GoogLeNet model is computationally complex, so it is not as easy to modify the number of channels as in VGG. In this section, we reduce the input height and width from 224 to 96 to simplify the computation. The changes in the shape of the output between the various modules is demonstrated below.

```{.python .input  n=7}
X = nd.random.uniform(shape=(1, 1, 96, 96))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

## Data Acquisition and Training

Here, we train a GoogLeNet model using images with a height and width of 96 pixels. The images used for training are again from the Fashion-MNIST data set.

```{.python .input  n=8}
lr, num_epochs, batch_size, ctx = 0.1, 5, 128, gb.try_gpu()
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size, resize=96)
gb.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)
```

## Summary

* The Inception block is equivalent to a subnetwork with four paths. It extracts information in parallel through convolutional layers of different window shapes and maximum pooling layers, and reduces the number of channels using the $1\times 1$ convolutional layer to reduce model complexity.

* GoogLeNet connects multiple well-designed Inception blocks with other layers in series. The ratio of the number of channels assigned in the Inception block is obtained through a large number of experiments on the ImageNet data set.

* GoogLeNet, as well as its succeeding versions, was one of the most efficient models on ImageNet, providing similar test accuracy with lower computational complexity.

## exercise

* There are several succeeding versions of GoogLeNet. Try to implement and run them and observe the experimental results. These subsequent versions either add a batch normalization layer (described later in this chapter)[2], make adjustments to the Inception block[3], or include it in the residual connection (described later in this chapter)[4].

* Compare the model parameter sizes of AlexNet, VGG, and NiN with GoogLeNet. How do the latter two network architectures significantly reduce the model parameter size?


## Scan the QR Code to Access [Discussions](https://discuss.gluon.ai/t/topic/1662)

![](../img/qr_googlenet.svg)

## References

[1] Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., & Anguelov, D. & Rabinovich, A. (2015). Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1-9).

[2] Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167.

[3] Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the inception architecture for computer vision. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 2818-2826).

[4] Szegedy, C., Ioffe, S., Vanhoucke, V., & Alemi, A. A. (2017, February). Inception-v4, inception-resnet and the impact of residual connections on learning. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 4, p. 12).
