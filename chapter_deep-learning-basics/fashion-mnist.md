# Image Classification Data Set (Fashion-MNIST)

Before introducing the implementation for softmax regression, we will first introduce a multi-class image classification data set. 
It will be used multiple times in later chapters to allow us to observe the difference between model accuracy and computational efficiency between comparison algorithms. The most commonly used image classification data set is the handwritten digit recognition data set MNIST [1]. However, most models have a classification accuracy of over 95% on MNIST. In order to more intuitively observe the differences between the algorithms, we will use a more complex data set Fashion-MNIST [2].

## Get Data Set

First, import the packages or modules required in this section.

```{.python .input}
%matplotlib inline
import gluonbook as gb
from mxnet.gluon import data as gdata
import sys
import time
```

Next, we will download this data set through Gluon's `data` package. The data is automatically retrieved from the Internet the first time it is called. We specify the acquisition of a training data set, or a testing data set by the parameter `train`. The test data set, also called the testing set, is only used to evaluate the performance of the model and is not used to train the model.

```{.python .input  n=23}
mnist_train = gdata.vision.FashionMNIST(train=True)
mnist_test = gdata.vision.FashionMNIST(train=False)
```

The number of images for each category in the training set and the testing set is 6,000 and 1,000, respectively. Because there are 10 categories, the number of examples for the training set and the testing set is 60,000 and 10,000, respectively.

```{.python .input}
len(mnist_train), len(mnist_test)
```

We can access any example by square brackets `[]`, and next, we will get the image and label of the first example.

```{.python .input  n=24}
feature, label = mnist_train[0]
```

The variable `feature` corresponds to an image with a height and width of 28 pixels. The value of each pixel is between 0 and 255 8-bit unsigned integer (uint8). It uses a 3D NDArray to store. Its last dimension is the number of channels. Since the data set is a grayscale image, the number of channels is 1. For the sake of simplicity, we will record the shape of the image with the height and width of $h$ and $w$ pixels, respectively, as $h \times w$ or `(h, w)`.

```{.python .input}
feature.shape, feature.dtype
```

The label of the image is represented by a scalar representation of NumPy. Its type is a 32-bit integer.

```{.python .input}
label, type(label), label.dtype
```

There are 10 categories in Fashion-MNIST: t-shirt, trousers, pullover, dress, coat, sandal, shirt, sneaker, bag and ankle boot. The following function can convert a numeric label into a corresponding text label.

```{.python .input  n=25}
# This function has been saved in the gluonbook package for future use.
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
```

The following defines a function that can draw multiple images and corresponding labels in a single line.

```{.python .input}
# This function has been saved in the gluonbook package for future use.
def show_fashion_mnist(images, labels):
    gb.use_svg_display()
    # Here _ means that we ignore (not use) variables.
    _, figs = gb.plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.reshape((28, 28)).asnumpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
```

Next, let's take a look at the image contents and text labels for the first nine examples in the training data set.

```{.python .input  n=27}
X, y = mnist_train[0:9]
show_fashion_mnist(X, get_fashion_mnist_labels(y))
```

## Read Mini-batch

We will train the model on the training dataset and evaluate the model's performance on the test dataset. Although we can define a function that reads a mini-batch of data examples by `yield` as in the section ["Linear Regression Implementation Starting from Scratch" ](linear-regression-scratch.md), for the sake of code simplicity, here we create the `DataLoader` instance directly. The instance reads a mini-batch of data with an example number of `batch_size` each time. The batch size `batch_size` here is a hyper-parameter.

In practice, data reading is often a performance bottleneck for training, especially when the model is simpler, or the computing hardware performance is high. A handy feature of Gluon's `DataLoader` is the ability to use multiple processes to speed up data reading (not currently supported on the Windows operating system). Here we set 4 processes to read the data by the parameter `num_workers`.

In addition, we convert the image data from the uint8 format to the 32-bit floating point format using the `ToTensor` class, and divide by 255 so that all pixels have values between 0 and 1. The `ToTensor` class also moves the image channel from the last dimension to the first dimension to facilitate the convolutional neural network calculations introduced later. Through the `transform_first` function of the data set, we apply the transformation of `ToTensor` to the first element of each data example (image and label), i.e., the image.

```{.python .input  n=28}
batch_size = 256
transformer = gdata.vision.transforms.ToTensor()
if sys.platform.startswith('win'):
    num_workers = 0  # 0 means no additional processes are needed to speed up the reading of data.
else:
    num_workers = 4

train_iter = gdata.DataLoader(mnist_train.transform_first(transformer),
                              batch_size, shuffle=True,
                              num_workers=num_workers)
test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),
                             batch_size, shuffle=False,
                             num_workers=num_workers)
```

The logic that we will use to obtain and read the Fashion-MNIST data set is encapsulated in the `gluonbook.load_data_fashion_mnist` function for calling in later chapters. This function will return two variables, `train_iter` and `test_iter`. As the content of this book continues to deepen, we will further improve this function. Its full implementation will be described in the section ["Deep Convolutional Neural Networks (AlexNet)"](../chapter_convolutional-neural-networks/alexnet.md).

Finally, we will look at the time it takes to read the training data.

```{.python .input}
start = time.time()
for X, y in train_iter:
    continue
'%.2f sec' % (time.time() - start)
```

## Summary

* Fashion-MNIST is an apparel classification data set containing 10 categories, which we will use to test the performance of different algorithms in later chapters.
* We will record the shape of the image with the height and width of $h$ and $w$ pixels, respectively, as $h \times w$ or `(h, w)`.

## exercise

* Does reducing `batch_size` (for instance, to 1) affect read performance?
* For non-Windows users, try modifying `num_workers` to see how it affects read performance.
* Viewing the MXNet documentation, what other data sets are available in `gdata.vision`?
* Viewing the MXNet documentation, what other transformation methods are available in `gdata.vision.transforms`?


## Scan the QR code to get to the [forum](https://discuss.gluon.ai/t/topic/7760)

![](../img/qr_fashion-mnist.svg)


## References

[1] LeCun, Y., Cortes, C., & Burges, C. http://yann.lecun.com/exdb/mnist/

[2] Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms. arXiv preprint arXiv:1708.07747.
