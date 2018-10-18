# Two-dimensional Convolutional Layers

A convolutional neural network is a neural network that contains convolutional layers. The convolutional neural networks used in this chapter use two-dimensional convolutional layers, the most common type. Such a layer has two spatial dimensions (height and width) and are commonly used to process image data. In this section, we will show how the simple form of the two-dimensional convolutional layer works.


## Two-dimensional Cross-correlation Operations

Although the convolutional layer is named after the convolution operation, we usually use a more intuitive cross-correlation operation in the convolutional layer. In a two-dimensional convolutional layer, a two-dimensional input array and a two-dimensional kernel array output a two-dimensional array through a cross-correlation operation.
Here, we use a specific example to explain the meaning of two-dimensional cross-correlation operations. As shown in Figure 5.1, the input is a two-dimensional array with a height of 3 and width of 3. We mark the shape of the array as $3 \times 3$ or (3, 3). The height and width of the kernel array are both 2. This array is also called a kernel or filter in convolution computations. The shape of the kernel window (also known as the convolution window) depends on the height and width of the kernel, which is $2 \times 2$.

![Two-dimensional cross-correlation operation. The shaded portions are the first output element and the input and kernel array elements used in its computation: $0\times0+1\times1+3\times2+4\times3=19$. ](../img/correlation.svg)

In the two-dimensional cross-correlation operation, the convolution window starts from the top-left of the input array, and slides in the input array from left to right and top to bottom. When the convolution window slides to a certain position, the input subarray in the window and kernel array are multiplied and summed by element to get the element at the corresponding location in the output array. The output array in Figure 5.1 has a height of 2 and width of 2, and the four elements are derived from a two-dimensional cross-correlation operation:

$$
0\times0+1\times1+3\times2+4\times3=19,\\
1\times0+2\times1+4\times2+5\times3=25,\\
3\times0+4\times1+6\times2+7\times3=37,\\
4\times0+5\times1+7\times2+8\times3=43.\\
$$

Next, we implement the above process in the `corr2d` function. It accepts the input array `X` with the kernel array `K` and outputs the array `Y`.

```{.python .input}
from mxnet import autograd, nd
from mxnet.gluon import nn

def corr2d(X, K):  # This function has been saved in the gluonbook package for future use.
    h, w = K.shape
    Y = nd.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y
```

We can construct the input array `X` and the kernel array `K` in Figure 5.1 to validate the output of the two-dimensional cross-correlation operation.

```{.python .input}
X = nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = nd.array([[0, 1], [2, 3]])
corr2d(X, K)
```

## Two-dimensional Convolutional Layers

The two-dimensional convolutional layer cross-correlates the input and kernels and adds a scalar bias to get the output. The model parameters of the convolutional layer include the kernel and scalar bias. When training the model, we usually randomly initialize the kernel and then continuously iterate the kernel and bias.

Next, we implement a custom two-dimensional convolutional layer based on the `corr2d` function. In the `__init__` constructor function, we declare `weight` and `bias` are two model parameters. The forward computation function `forward` directly calls the `corr2d` function and adds the bias.

```{.python .input  n=70}
class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))

    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()
```

The convolutional layer with a convolution window shape of $p \times q$ is called the $p \times q$ convolutional layer. Similarly, the $p \times q$ convolution or $p \times q$ kernel states that the height and width of the kernel are $p$ and $q$, respectively.


## Object Edge Detection in Images

Now, we will look at a simple application of a convolutional layer: detecting the edge of an object in an image, that is, finding the location of the pixel change. First, we construct an image of $6\times 8$ (image with height and width of 6 and 8 pixels respectively). The middle four columns are black (0), and the rest are white (1).

```{.python .input  n=66}
X = nd.ones((6, 8))
X[:, 2:6] = 0
X
```

Then, we construct a kernel `K` with a height and width of 1 and 2. When this performs the cross-correlation operation with the input, if the horizontally adjacent elements are the same, the output is 0. Otherwise, the output is non-zero.

```{.python .input  n=67}
K = nd.array([[1, -1]])
```

Enter `X` and our designed kernel `K` to perform the cross-correlation operations. As you can see, we will detect 1 for the edge from white to black and -1 for the edge from black to white. The rest of the outputs are 0.

```{.python .input  n=69}
Y = corr2d(X, K)
Y
```

From this, we can see that the convolutional layer can effectively characterize the local space by reusing the kernel.


## Learning a Kernel Array Through Data

Finally, we will look at an example that uses input data `X` and output data `Y` in object edge detection to learn the kernel array `K` we constructed. We first construct a convolutional layer and initialize its kernel into a random array. Next, in each iteration, we use the square error to compare `Y` and the output of the convolutional layer, then calculate the gradient to update the weight. For the sake of simplicity, the convolutional layer here ignores the bias.

We have previously constructed the `Conv2D` class. However, because `corr2d` performs assignment to a single element (`[i, j]=`), it fails to automatically find the gradient. Below, we use the `Conv2D` class provided by Gluon to implement this example.

```{.python .input  n=83}
# Construct a convolutional layer with 1 output channel (channels will be introduced in the following section) and a kernel array shape of (1, 2).
conv2d = nn.Conv2D(1, kernel_size=(1, 2))
conv2d.initialize()

# The two-dimensional convolutional layer uses four-dimensional input and output in the format of (example channel, height, width), where the batch size (number of examples in the batch) and 
# the number of channels are both 1.
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))

for i in range(10):
    with autograd.record():
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
    l.backward()
    # For the sake of simplicity, we ignore the bias here.
    conv2d.weight.data()[:] -= 3e-2 * conv2d.weight.grad()
    if (i + 1) % 2 == 0:
        print('batch %d, loss %.3f' % (i + 1, l.sum().asscalar()))
```

As you can see, the error has dropped to a relatively small value after 10 iterations. Now we will take a look at the kernel array we learned.

```{.python .input}
conv2d.weight.data().reshape((1, 2))
```

We find that the kernel array we learned is similar to the kernel array `K` we defined earlier.

## Cross-correlation and Convolution Operations

In fact, the convolution operation is similar to the cross-correlation operation. In order to get the output of the convolution operation, we simply need to flip the kernel array left to right and upside-down, and then perform the cross-correlation operation with the input array. As you can see, the convolution operation and cross-correlation operation are similar, but if they use the same kernel array, the output is often different with the same input.

So, you might wonder why convolutional layers can use the cross-correlation operation instead of the convolution operation. In fact, kernel arrays are learned in deep learning: the convolutional layer does not affect the output of model prediction, whether it uses a cross-correlation operation or convolution operation. To explain this, assume that the convolutional layer uses the cross-correlation operation to learn the kernel array in Figure 5.1. If other conditions are not changed, the kernel array learned by the convolution operation, that is, the kernel array in Figure 5.1, is flipped upside down, left to right. That is to say, when we perform the convolution operation once again on the input in Figure 5.1 and the learned flipped kernel array, the output in Figure 5.1 is still obtained. In accordance with the standard terminology in deep learning literature, the convolution operations mentioned in this book refer to cross-correlation operations, unless otherwise stated.


## Summary

* The core computation of a two-dimensional convolutional layer is a two-dimensional cross-correlation operation. In its simplest form, this performs a cross-correlation operation on the two-dimensional input data and the kernel, and then adds a bias.
* We can design a kernel to detect edges in images.
* We can learn the kernel through data.


## exercise

* Construct an input image `X` which has horizontal edges. How do you design a kernel `K` to detect horizontal edges in an image? What if the edges are diagonal?
* When you try to automatically find the gradient for the `Conv2D` class we created, what kind of error message do you see? In the `forward` function of this class, the `corr2d` function is replaced with the `nd.Convolution` class, which makes it possible to automatically find the gradient.
* How do you represent a cross-correlation operation as a matrix multiplication by changing the input and kernel arrays?
* How do you construct a fully connected layer for object edge detection?

## Scan the QR Code to Access [Discussions](https://discuss.gluon.ai/t/topic/6314)

![](../img/qr_conv-layer.svg)
