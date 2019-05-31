# Transposed Convolution
:label:`chapter_transposed_conv`

The layers we introduced so far for convolutional neural networks, including 
convolutional layers (:numref:`chapter_conv_layer`) and pooling layers (:numref:`chapter_pooling`), often reducethe input width and height, or keep them unchanged. Applications such as semantic segmentation (:numref:`chapter_semantic_segmentation`) and generative adversarial networks (:numref:`chapter_dcgan`), however, require to predict values for each pixel and therefore needs to increase input width and height. Transposed convolution, often named fractionally-strided convolution or deconvolution, serves this purpose.

```{.python .input  n=18}
from mxnet import nd, init
from mxnet.gluon import nn
```

## Basic 2D Transposed Convolution

Let's consider a basic case that both input and output channels are 1, with 0 padding and 1 stride. :numref:`fig_trans_conv` illustrates how transposed convolution with a $2\times 2$ kernel is computed on the $2\times 2$ input matrix. 

![Transposed convolution layer with a $2\times 2$ kernel.](../img/trans_conv.svg)
:label:`fig_trans_conv`

We can implement this operation by giving matrix kernel $K$ and matrix input $X$. 

```{.python .input  n=20}
def trans_conv(X, K):
    h, w = K.shape
    Y = nd.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i: i + h, j: j + w] += X[i, j] * K
    return Y
```

Remember the convolution computes results by `Y[i, j] = (X[i: i + h, j: j + w] * K).sum()` (refer to `corr2d` in :numref:`chapter_conv_layer`), which summarizes input values through the kernel. While the transposed convolution broadcasts input values through the kernel, which results in a larger output shape. 

Verify the results in :numref:`fig_trans_conv`.

```{.python .input  n=21}
X = nd.array([[0,1], [2,3]])
K = nd.array([[0,1], [2,3]])
trans_conv(X, K)
```

```{.json .output n=21}
[
 {
  "data": {
   "text/plain": "\n[[ 0.  0.  1.]\n [ 0.  4.  6.]\n [ 4. 12.  9.]]\n<NDArray 3x3 @cpu(0)>"
  },
  "execution_count": 21,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Or we can use `nn.Conv2DTranspose` to obtain the same results. As `nn.Conv2D`, both input and kernel should be 4-D tensors. 

```{.python .input  n=31}
X, K = X.reshape((1, 1, 2, 2)),  K.reshape((1, 1, 2, 2))
tconv = nn.Conv2DTranspose(1, kernel_size=2)
tconv.initialize(init.Constant(K))
tconv(X)
```

```{.json .output n=31}
[
 {
  "data": {
   "text/plain": "\n[[[[ 0.  0.  1.]\n   [ 0.  4.  6.]\n   [ 4. 12.  9.]]]]\n<NDArray 1x1x3x3 @cpu(0)>"
  },
  "execution_count": 31,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## Padding, Strides, and Channels

We apply padding elements to the input in convolution, while they are applied to the output in transposed convolution. A $1\times 1$ padding means we first compute the output as normal, then remove the first/last rows and columns. 

```{.python .input  n=33}
tconv = nn.Conv2DTranspose(1, kernel_size=2, padding=1)
tconv.initialize(init.Constant(K))
tconv(X)
```

```{.json .output n=33}
[
 {
  "data": {
   "text/plain": "\n[[[[4.]]]]\n<NDArray 1x1x1x1 @cpu(0)>"
  },
  "execution_count": 33,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Similarly, strides are applied to outputs as well.

```{.python .input  n=35}
tconv = nn.Conv2DTranspose(1, kernel_size=2, strides=2)
tconv.initialize(init.Constant(K))
tconv(X)
```

```{.json .output n=35}
[
 {
  "data": {
   "text/plain": "\n[[[[0. 0. 0. 1.]\n   [0. 0. 2. 3.]\n   [0. 2. 0. 3.]\n   [4. 6. 6. 9.]]]]\n<NDArray 1x1x4x4 @cpu(0)>"
  },
  "execution_count": 35,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

But multi-channels work the same as convolution. As a result, if we feed $X$ into a convolutional layer $f$ to compute $Y=f(X)$ and create a transposed convolution layer $g$ with the same hyper-parameters as $f$ except for the output channel set to be the channel size of $X$, then $g(Y)$ should has the same shape as $X$. Let's verify this statement. 

```{.python .input  n=37}
X = nd.random.uniform(shape=(1, 10, 16, 16))
conv = nn.Conv2D(20, kernel_size=5, padding=2, strides=3)
tconv = nn.Conv2DTranspose(10, kernel_size=5, padding=2, strides=3)
conv.initialize()
tconv.initialize()
tconv(conv(X)).shape == X.shape
```

```{.json .output n=37}
[
 {
  "data": {
   "text/plain": "True"
  },
  "execution_count": 37,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## Relation to mm

```{.python .input  n=5}
X = nd.arange(1, 17).reshape((1, 1, 4, 4))
K = nd.arange(1, 10).reshape((1, 1, 3, 3))
conv = nn.Conv2D(channels=1, kernel_size=3)
conv.initialize(init.Constant(K))
conv(X), K
```

```{.json .output n=5}
[
 {
  "data": {
   "text/plain": "(\n [[[[348. 393.]\n    [528. 573.]]]]\n <NDArray 1x1x2x2 @cpu(0)>, \n [[[[1. 2. 3.]\n    [4. 5. 6.]\n    [7. 8. 9.]]]]\n <NDArray 1x1x3x3 @cpu(0)>)"
  },
  "execution_count": 5,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Next, we rewrite convolution kernel `K` as a sparse matrix `W` with a large number of zero elements, i.e. a weight matrix. The shape of the weight matrix is (4,16), where the non-zero elements are taken from the elements in convolution kernel `K`. Enter `X` and concatenate line by line to get a vector of length 16. Then, perform matrix multiplication for `W` and the `X` vector to get a vector of length 4. After the transformation, we can get the same result as the convolution operation above. As you can see, in this example, we implement the convolution operation using matrix multiplication.

```{.python .input  n=6}
W, k = nd.zeros((4, 16)), nd.zeros(11)
k[:3], k[4:7], k[8:] = K[0, 0, 0, :], K[0, 0, 1, :], K[0, 0, 2, :]
W[0, 0:11], W[1, 1:12], W[2, 4:15], W[3, 5:16] = k, k, k, k
nd.dot(W, X.reshape(16)).reshape((1, 1, 2, 2)), W
```

```{.json .output n=6}
[
 {
  "data": {
   "text/plain": "(\n [[[[348. 393.]\n    [528. 573.]]]]\n <NDArray 1x1x2x2 @cpu(0)>, \n [[1. 2. 3. 0. 4. 5. 6. 0. 7. 8. 9. 0. 0. 0. 0. 0.]\n  [0. 1. 2. 3. 0. 4. 5. 6. 0. 7. 8. 9. 0. 0. 0. 0.]\n  [0. 0. 0. 0. 1. 2. 3. 0. 4. 5. 6. 0. 7. 8. 9. 0.]\n  [0. 0. 0. 0. 0. 1. 2. 3. 0. 4. 5. 6. 0. 7. 8. 9.]]\n <NDArray 4x16 @cpu(0)>)"
  },
  "execution_count": 6,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Now we will describe the convolution operation from the perspective of matrix multiplication. Let the input vector be $\boldsymbol{x}$ and weight matrix be $\boldsymbol{W}$. The implementation of the convolutional forward computation function can be considered as the multiplication of the function input by the weight matrix to output the vector $\boldsymbol{ y} = \boldsymbol{W}\boldsymbol{x}$. We know that back propagation needs to be based on chain rules. Because $\nabla_{\boldsymbol{x}} \boldsymbol{y} = \boldsymbol{W}^\top$, the implementation of the convolutional back propagation function can be considered as the multiplication of the function input by the transposed weight matrix $\boldsymbol{W}^\top$. The transposed convolution layer exchanges the forward computation function and the back propagation function of the convolution layer. These two functions can be regarded as the multiplication of the function input vectors by $\boldsymbol{W}^\top$ and $\boldsymbol{W}$, respectively.

It is not difficult to see that the transposed convolution layer can be used to exchange the shape of input and output of the convolution layer. Let us continue to describe convolution using matrix multiplication. Let the weight matrix be a matrix with a shape of $4\times16$. For an input vector of length 16, the convolution forward computation outputs a vector with a length of 4. If the length of the input vector is 4 and the shape of the transpose weight matrix is $16\times4$, then the transposed convolution layer outputs a vector of length 16. In model design, transposed convolution layers are often used to transform smaller feature maps into larger ones. In a full convolutional network, when the input is a feature map with a high height and a wide width, the transposed convolution layer can be used to magnify the height and width to the size of the input image.

Now we will look at an example. Construct a convolution layer `conv` and let shape of input `X` be (1,3,64,64). The number of channels for convolution output `Y` is increased to 10, but the height and width are reduced by half.

```{.python .input  n=7}
conv = nn.Conv2D(10, kernel_size=4, padding=1, strides=2)
conv.initialize()

X = nd.random.uniform(shape=(1, 3, 64, 64))
Y = conv(X)
Y.shape
```

```{.json .output n=7}
[
 {
  "data": {
   "text/plain": "(1, 10, 32, 32)"
  },
  "execution_count": 7,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Next, we construct transposed convolution layer `conv_trans` by creating a `Conv2DTranspose` instance. Here, we assume the convolution kernel shape, padding, and stride of `conv_trans` are the same with those in `conv`, and the number of output channels is 3. When the input is output `Y` of the convolution layer `conv`, the transposed convolution layer output has the same height and width as convolution layer input. The transposed convolution layer magnifies the height and width of the feature map by a factor of 2.

```{.python .input  n=8}
conv_trans = nn.Conv2DTranspose(3, kernel_size=4, padding=1, strides=2)
conv_trans.initialize()
conv_trans(Y).shape
```

```{.json .output n=8}
[
 {
  "data": {
   "text/plain": "(1, 3, 64, 64)"
  },
  "execution_count": 8,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

In the literature, transposed convolution is also sometimes referred to as
fractionally-strided convolution :ref:`Dumoulin.Visin.2016`
