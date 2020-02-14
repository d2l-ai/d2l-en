# Fully Convolutional Networks (FCN)
:label:`sec_fcn`

We previously discussed semantic segmentation using each pixel in an image for
category prediction. A fully convolutional network (FCN)
:cite:`Long.Shelhamer.Darrell.2015` uses a convolutional neural network to
transform image pixels to pixel categories. Unlike the convolutional neural
networks previously introduced, an FCN transforms the height and width of the
intermediate layer feature map back to the size of input image through the
transposed convolution layer, so that the predictions have a one-to-one
correspondence with input image in spatial dimension (height and width). Given a
position on the spatial dimension, the output of the channel dimension will be a
category prediction of the pixel corresponding to the location.

We will first import the package or module needed for the experiment and then
explain the transposed convolution layer.

```{.python .input  n=1}
%matplotlib inline
import d2l
from mxnet import gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()
```

## Constructing a Model

Here, we demonstrate the most basic design of a fully convolutional network model. As shown in :numref:`fig_fcn`, the fully convolutional network first uses the convolutional neural network to extract image features likes an object detection model, then transforms the number of channels into the number of categories through the $1\times 1$ convolution layer, and finally transforms the height and width of the feature map to the size of the input image by using the transposed convolution layer :numref:`sec_transposed_conv`. The model output has the same height and width as the input image and has a one-to-one correspondence in spatial positions. The final output channel contains the category prediction of the pixel of the corresponding spatial position.

![Fully convolutional network. ](../img/fcn.svg)
:label:`fig_fcn`

Similar to FPN :numref:`sec_fpn`, we use a ResNet-18 model pre-trained on the ImageNet dataset to extract image features. The feature maps before the global average pooling layers are kept. 

```{.python .input  n=2}
net = d2l.detection_backbone()
```

Given an input of a height and width of 320 and 480 respectively, the forward computation of `net` will reduce the height and width of the input to $1/32$ of the original, i.e., 10 and 15.

```{.python .input  n=4}
X = np.random.uniform(size=(1, 3, 320, 480))
net(X).shape
```

Next, we transform the number of output channels to the number of categories of
Pascal VOC2012 (21) through the $1\times 1$ convolution layer. Finally, we need
to magnify the height and width of the feature map by a factor of 32 to change
them back to the height and width of the input image. Recall the calculation
method for the convolution layer output shape described in
:numref:`sec_padding`. Because
$(320-64+16\times2+32)/32=10$ and $(480-64+16\times2+32)/32=15$, we construct a
transposed convolution layer with a stride of 32 and set the height and width of
the convolution kernel to 64 and the padding to 16. It is not difficult to see
that, if the stride is $s$, the padding is $s/2$ (assuming $s/2$ is an integer),
and the height and width of the convolution kernel are $2s$, the transposed
convolution kernel will magnify both the height and width of the input by a
factor of $s$.

```{.python .input  n=5}
num_classes = 21
net.add(nn.Conv2D(num_classes, kernel_size=1),
        nn.Conv2DTranspose(
            num_classes, kernel_size=64, padding=16, strides=32))
```

## Initializing the Transposed Convolution Layer

We already know that the transposed convolution layer can magnify a feature map. In image processing, sometimes we need to magnify the image, i.e., upsampling. There are many methods for upsampling, and one common method is bilinear interpolation. Simply speaking, in order to get the pixel of the output image at the coordinates $(x, y)$, the coordinates are first mapped to the coordinates of the input image $(x', y')$. This can be done based on the ratio of the size of three input to the size of the output. The mapped values $x'$ and $y'$ are usually real numbers. Then, we find the four pixels closest to the coordinate $(x', y')$ on the input image. Finally, the pixels of the output image at coordinates $(x, y)$ are calculated based on these four pixels on the input image and their relative distances to $(x', y')$. Upsampling by bilinear interpolation can be implemented by transposed convolution layer of the convolution kernel constructed using the following `bilinear_kernel` function. Due to space limitations, we only give the implementation of the `bilinear_kernel` function and will not discuss the principles of the algorithm.

```{.python .input  n=6}
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (np.arange(kernel_size).reshape(-1, 1),
          np.arange(kernel_size).reshape(1, -1))
    filt = (1 - np.abs(og[0] - center) / factor) * \
           (1 - np.abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return np.array(weight)
```

Now, we will experiment with bilinear interpolation upsampling implemented by transposed convolution layers. Construct a transposed convolution layer that magnifies height and width of input by a factor of 2 and initialize its convolution kernel with the `bilinear_kernel` function.

```{.python .input  n=7}
conv_trans = nn.Conv2DTranspose(3, kernel_size=4, padding=1, strides=2)
conv_trans.initialize(init.Constant(bilinear_kernel(3, 3, 4)))
```

Read the image `X` and record the result of upsampling as `Y`. In order to print the image, we need to adjust the position of the channel dimension.

```{.python .input  n=8}
img = image.imread(d2l.download('catdog'))
X = np.expand_dims(img.astype('float32').transpose(2, 0, 1), axis=0) / 255
Y = conv_trans(X)
out_img = Y[0].transpose(1, 2, 0)
```

As you can see, the transposed convolution layer magnifies both the height and width of the image by a factor of 2. It is worth mentioning that, besides to the difference in coordinate scale, the image magnified by bilinear interpolation and original image printed in :numref:`sec_bbox` look the same.

```{.python .input  n=9}
d2l.set_figsize((3.5, 2.5))
print('input image shape:', img.shape)
d2l.plt.imshow(img.asnumpy());
print('output image shape:', out_img.shape)
d2l.plt.imshow(out_img.asnumpy());
```

In a fully convolutional network, we initialize the transposed convolution layer for upsampled bilinear interpolation. For a $1\times 1$ convolution layer, we use Xavier for randomly initialization.

```{.python .input  n=10}
W = bilinear_kernel(num_classes, num_classes, 64)
net[-1].initialize(init.Constant(W))
net[-2].initialize(init=init.Xavier())
```

## Reading the Dataset

We read the dataset using the method described in the previous section. Here, we specify shape of the randomly cropped output image as $320\times 480$, so both the height and width are divisible by 32.

```{.python .input  n=11}
batch_size, height, width = 32, 320, 480
train_iter, test_iter = d2l.load_data_voc_segmentation(batch_size, height, width)
```

## Training

Now we can start training the model. The loss function and accuracy calculation here are not substantially different from those used in image classification. Because we use the channel of the transposed convolution layer to predict pixel categories, the `axis=1` (channel dimension) option is specified in `SoftmaxCrossEntropyLoss`. In addition, the model calculates the accuracy based on whether the prediction category of each pixel is correct.

```{.python .input  n=12}
num_epochs, lr, wd, ctx = 5, 0.1, 1e-3, d2l.try_all_gpus()
loss = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)
net.collect_params().reset_ctx(ctx)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': lr, 'wd': wd})
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, ctx)
```

## Prediction

To visualize the predicted categories for each pixel, we define a function to map the predicted categories back to their labeled colors.

```{.python .input  n=25}
def label2image(pred):
    colormap = np.array(d2l.VOC_SEG_COLORS, dtype='uint8')
    idx = pred.astype('int32').as_in_context(colormap.context)
    return colormap[idx, :]
```

Let's read a batch from the test dataset, and predict the labels.

```{.python .input}
for X, Y in test_iter:
    break
    
preds = net(X.as_in_context(ctx[0])).argmax(axis=1)
```

Now we can visualize the first 4 predictions (in middle row) with the original images (in top row) and ground truth labels (in bottle row). 


```{.python .input  n=44}
n = 4
images = X[:n].transpose(0,2,3,1)*d2l.RGB_STD+d2l.RGB_MEAN
images = (images*255).astype('uint8')
d2l.show_images(np.concatenate((images, label2image(preds[:n]), 
                                label2image(Y[:n]))), 3, n);
```

## Summary

* The fully convolutional network first uses the convolutional neural network to extract image features, then transforms the number of channels into the number of categories through the $1\times 1$ convolution layer, and finally transforms the height and width of the feature map to the size of the input image by using the transposed convolution layer to output the category of each pixel.
* In a fully convolutional network, we initialize the transposed convolution layer for upsampled bilinear interpolation.


## Exercises

1. If we use Xavier to randomly initialize the transposed convolution layer, what will happen to the result?
1. Can you further improve the accuracy of the model by tuning the hyper-parameters?
1. The outputs of some intermediate layers of the convolutional neural network are also used in the paper on fully convolutional networks. Try to implement this idea.


## [Discussions](https://discuss.mxnet.io/t/2454)

![](../img/qr_fcn.svg)
