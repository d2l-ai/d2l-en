# GPU Computing

So far, we have been using CPU computing. For complex neural networks and large-scale data, using only CPUs for computation may be inefficient. In this section, we will discuss how to use a single Nvidia GPU for calculations. First, make sure you have at least one Nvidia GPU installed. Then, download CUDA and follow the prompts to set the appropriate path[1]. Once these preparations are complete, the `nvidia-smi` command can be used to view the graphics card information.

```{.python .input  n=1}
!nvidia-smi
```

Next, we need to confirm that the GPU version of MXNet is installed. If a CPU version of MXNet is already installed, we need to uninstall it first. For example, use the `pip uninstall mxnet` command, then install the corresponding MXNet version according to the CUDA version. Assuming you have CUDA 9.0 installed, you can install the MXNet version that supports CUDA 9.0 by `pip install mxnet-cu90`. To run the programs in this section, you need at least two GPUs.

## Computing Devices

MXNet can specify devices, such as CPUs and GPUs, for storage and calculation. By default, MXNet creates data in the main memory and then uses the CPU to calculate it. In MXNet, the CPU and GPU can be indicated by `cpu()` and `gpu()`. It should be noted that `mx.cpu()` (or any integer in the parentheses) means all physical CPUs and memory. This means that MXNet's calculations will try to use all CPU cores. However, `mx.gpu()` only represents one graphic card and the corresponding graphic memory. If there are multiple GPUs, we use `mx.gpu(i)` to represent the $i$-th GPU ($i$ starts from 0). Also, `mx.gpu(0)` and `mx.gpu()` are equivalent.

```{.python .input}
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn

mx.cpu(), mx.gpu(), mx.gpu(1)
```

## GPU calculation for NDArray

By default, NDArray exists on the CPU. Therefore, we will see the `@cpu(0)` identifier each time we print NDArray.

```{.python .input  n=4}
x = nd.array([1, 2, 3])
x
```

We can use the `context` property of NDArray to view the device where the NDArray is located.

```{.python .input}
x.context
```

### Storage on the GPU

There are several ways to store NDArray on the GPU. For example, we can specify a storage device with the `ctx` parameter when creating an NDArray. Next, we create the NDArray variable `a` on `gpu(0)`. Notice that when printing `a`, the device information becomes `@gpu(0)`. The NDArray created on a GPU only consumes the memory of this GPU. We can use the `nvidia-smi` command to view GPU memory usage. In general, we need to make sure we do not create data that exceeds the GPU memory limit.

```{.python .input  n=5}
a = nd.array([1, 2, 3], ctx=mx.gpu())
a
```

Assuming you have at least two GPUs, the following code will create a random array on `gpu(1)`.

```{.python .input}
b = nd.random.uniform(shape=(2, 3), ctx=mx.gpu(1))
b
```

In addition to that specified at the time of creation, we can also transfer data between devices through the `copyto` and `as_in_context` functions. Next, we copy the NDArray variable `x` on the CPU to `gpu(0)`.

```{.python .input  n=7}
y = x.copyto(mx.gpu())
y
```

```{.python .input}
z = x.as_in_context(mx.gpu())
z
```

It is important to note that, if the `context` of the source variable and the target variable are consistent, then the `as_in_context` function causes the target variable and the source variable to share the memory of the source variable.

```{.python .input  n=8}
y.as_in_context(mx.gpu()) is y
```

The `copyto` function always creates new memory for the target variable.

```{.python .input}
y.copyto(mx.gpu()) is y
```

### Computing on the GPU

MXNet calculations are performed on the device specified by the data `context`. In order to use GPU computing, we only need to store the data on the GPU in advance. The results of the calculation are automatically saved on the same GPU.

```{.python .input  n=9}
(z + 2).exp() * y
```

Note that MXNet requires all input data for calculation to be on the CPU or the same GPU. It is designed this way because data interaction between the CPU and different GPUs is usually time consuming. Therefore, MXNet expects the user to specify that the input data for calculation is on the CPU or the same GPU. For example, if you use the NDArray variable `x` on the CPU and the NDArray variable `y` for operation, then an error message will appear. When we print NDArray or convert NDArray to NumPy format, if the data is not in main memory, MXNet will copy it to the main memory first, resulting in additional transmission overhead.

## Gluon's GPU computing

Similar to NDArray, Gluon's model can specify devices through the `ctx` parameter during initialization. The following code initializes the model parameters on the GPU.

```{.python .input  n=12}
net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(ctx=mx.gpu())
```

When the input is an NDArray on the GPU, Gluon will calculate the result on the same GPU.

```{.python .input  n=13}
net(y)
```

Next, let we confirm that the model parameters are stored on the same GPU.

```{.python .input  n=14}
net[0].weight.data()
```

## Summary

* MXNet can specify devices for storage and calculation, such as CPU or GPU. By default, MXNet creates data in the main memory and then uses the CPU to calculate it.
* MXNet requires all input data for calculation to be on the CPU or the same GPU.

## exercise

* Try a larger computation task, such as the multiplication of large matrices, and see the difference in speed between the CPU and GPU. What about a task with a small amount of calculations?
* How should we read and write model parameters on the GPU?

## Scan the QR Code to Access [Discussions](https://discuss.gluon.ai/t/topic/988)

![](../img/qr_use-gpu.svg)


## References

[1] CUDA download address. https://developer.nvidia.com/cuda-downloads
