# Padding and Stride

In the example in the previous section, we used an input with a height and width of 3 and a convolution kernel with a height and width of 2 to get an output with a height and a width of 2. In general, assuming the input shape is $n_h\times n_w$ and the convolution kernel window shape is $k_h\times k_w$, then the output shape will be

$(n_h-k_h+1) \times (n_w-k_w+1).$

Therefore, the output shape of the convolutional layer is determined by the shape of the input and the shape of the convolution kernel window. In this section, we will introduce the two hyper-parameters of the convolutional layer, padding and stride. They can change the output shape for an input and convolution kernel of a given shape.

## Padding

Padding refers to padding elements (usually 0 elements) added on both sides of the input height and width. In Figure 5.2, we add an element with a value of 0 on both sides of the original input width and height, causing the input height and width to change from 3 to 5, and causing the output height and width to increase from 2 to 4.

![A two-dimensional cross-correlation computation with 0 elements padded on both sides of the input height and width. The shaded portions are the input and kernel array elements used by the first output element and its computations: $0\times0+0\times1+0\times2+0\times3=0$. ](../img/conv_pad.svg)

In general, if a total of $p_h$ rows are padded on both sides of the height and a total of $p_w$ columns are padded on both sides of width, the output shape will be

$(n_h-k_h+p_h+1)\times(n_w-k_w+p_w+1),$

This means that the height and width of the output will increase by $p_h$ and $p_w$ respectively.

In many cases, we will want to set $p_h=k_h-1$ and $p_w=k_w-1$ to give the input and output the same height and width. This will make it easier to predict the output shape of each layer when constructing the network. Assuming that $k_h$ is odd here, we will pad $p_h/2$ rows on both sides of the height. If $k_h$ is even, one possibility is padding $\lceil p_h/2\rceil$ rows on the top of the input and padding $\lfloor p_h/2\rfloor$ rows on the bottom. We will pad both sides of the width in the same way.

Convolutional neural networks often use convolution kernels with odd height and width values, such as 1, 3, 5, and 7, so the number of padding rows or columns on both sides are the same. For any two-dimensional array `X`, assume the element in its `i`th row and `j`th column is `X[i,j]`. When number of padding rows or columns on both sides are the same so that the input and output have the same height and width, we know that the output `Y[i,j]` is calculated by cross-correlation of the input and convolution kernel with the window centered on `X[i,j]`.

In the following example we create a two-dimensional convolutional layer with a height and width of 3, and then assume the padding number on both sides of the input height and width is 1. Given an input with a height and width of 8, we find that the height and width of the output is also 8.

```{.python .input  n=1}
from mxnet import nd
from mxnet.gluon import nn

# Now we define a convenience function to calculate the convolutional layer. This function initializes the convolutional layer weights and performs corresponding dimensionality elevations and reductions on the input and output.
def comp_conv2d(conv2d, X):
    conv2d.initialize()
    # (1,1) indicates that the batch size and the number of channels (described in later chapters) are both 1.
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])  # Exclude the first two dimensions that do not interest us: batch and channel.

# Note that here 1 row or column is padded on either side, so a total of 2 rows or columns are padded on the two sides.
conv2d = nn.Conv2D(1, kernel_size=3, padding=1)
X = nd.random.uniform(shape=(8, 8))
comp_conv2d(conv2d, X).shape
```

When the height and width of the convolution kernel are different, we can make the output and input have the same height and width by setting different padding numbers for height and width.

```{.python .input  n=2}
# Here, we use a convolution kernel with a height of 5 and a width of 3. The padding numbers on both sides of the height and width are 2 and 1, respectively.
conv2d = nn.Conv2D(1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape
```

## Stride

In the previous section, we introduced two-dimensional cross-correlation operations. The convolution window starts from the top-left of the input array, and slides in the input array from left to right and top to bottom. We refer to the number of rows and columns per slide as the stride.

In the current example, the stride is 1 in both the height and width directions. We can also use a larger stride. Figure 5.3 shows a two-dimensional cross-correlation operation with a stride of 3 on the height and 2 on the width. We can see that when the second element of the first column is output, the convolution window slides down three rows. The convolution window slides two columns to the right when the second element of the first row is output. When the convolution window slides two columns to the right on the input, there is no output result because the input element cannot fill the window.

![A two-dimensional cross-correlation operation with strides on height and width of 3 and 2. The shaded portions are the output element and the input and core array elements used in its computation: $0\times0+0\times1+1\times2+2\times3=8$, $0\times0+6\times1+0\times2+0\times3=6$. ](../img/conv_stride.svg)

In general, when the stride on the height is $s_h$ and the stride on the width is $s_w$, the output shape is

$\lfloor(n_h-k_h+p_h+s_h)/s_h\rfloor \times \lfloor(n_w-k_w+p_w+s_w)/s_w\rfloor.$

If we set $p_h=k_h-1$ and $p_w=k_w-1$, then the output shape will be simplified to $\lfloor(n_h+s_h-1)/s_h\rfloor \times \lfloor(n_w+s_w-1)/s_w\rfloor$. Going a step further, if the input height and width are divisible by the strides on the height and width, then the output shape will be $(n_h/s_h) \times (n_w/s_w)$.

Below, we set the strides on both the height and width to 2, thus halving the input height and width.

```{.python .input}
conv2d = nn.Conv2D(1, kernel_size=3, padding=1, strides=2)
comp_conv2d(conv2d, X).shape
```

Next, we will look at a slightly more complicated example.

```{.python .input  n=3}
conv2d = nn.Conv2D(1, kernel_size=(3, 5), padding=(0, 1), strides=(3, 4))
comp_conv2d(conv2d, X).shape
```

For the sake of brevity, when the padding number on both sides of the input height and width are $p_h$ and $p_w$ respectively, we call the padding $(p_h, p_w)$. Specifically, when $p_h = p_w = p$, the padding is $p$. When the strides on the height and width are $s_h$ and $s_w$, respectively, we call the stride $(s_h, s_w)$. Specifically, when $s_h = s_w = s$, the stride is $s$. By default, the padding is 0 and the stride is 1.



## Summary

* Padding can increase the height and width of the output. This is often used to give the output the same height and width as the input.
* The stride can reduce the height and width of the output, for example reducing the height and width of the output to only $1/n$ of the height and width of the input ($n$ is an integer greater than 1).

## exercise

* For the last example in this section, use the shape calculation formula to calculate the output shape to see if it is consistent with the experimental results.
* Try other padding and stride combinations on the experiments in this section.

## Scan the QR Code to Access [Discussions](https://discuss.gluon.ai/t/topic/6404)

![](../img/qr_padding-and-strides.svg)
