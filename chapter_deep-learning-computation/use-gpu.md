# GPUs
:label:`sec_use_gpu`

In the introduction, we discussed the rapid growth
of computation over the past two decades.
In a nutshell, GPU performance has increased
by a factor of 1000 every decade since 2000.
This offers great opportunity but it also suggests
a significant need to provide such performance.

|Decade|Dataset|Memory|Floating Point Calculations per Second|
|:--|:-|:-|:-|
|1970|100 (Iris)|1 KB|100 KF (Intel 8080)|
|1980|1 K (House prices in Boston)|100 KB|1 MF (Intel 80186)|
|1990|10 K (optical character recognition)|10 MB|10 MF (Intel 80486)|
|2000|10 M (web pages)|100 MB|1 GF (Intel Core)|
|2010|10 G (advertising)|1 GB|1 TF (NVIDIA C2050)|
|2020|1 T (social network)|100 GB|1 PF (NVIDIA DGX-2)|

In this section, we begin to discuss how to harness
this compute performance for your research.
First by using single GPUs and at a later point,
how to use multiple GPUs and multiple servers (with multiple GPUs).

In this section, we will discuss how
to use a single NVIDIA GPU for calculations.
First, make sure you have at least one NVIDIA GPU installed.
Then, [download CUDA](https://developer.nvidia.com/cuda-downloads)
and follow the prompts to set the appropriate path.
Once these preparations are complete,
the `nvidia-smi` command can be used
to view the graphics card information.

```{.python .input}
#@tab all
!nvidia-smi
```

:begin_tab:`mxnet`
You might have noticed that MXNet tensor
looks almost identical to NumPy.

But there are a few crucial differences.
One of the key features that distinguishes MXNet
from NumPy is its support for diverse hardware devices.

In MXNet, every array has a context.
So far, by default, all variables
and associated computation
have been assigned to the CPU.
Typically, other contexts might be various GPUs.
Things can get even hairier when
we deploy jobs across multiple servers.
By assigning arrays to contexts intelligently,
we can minimize the time spent
transferring data between devices.
For example, when training neural networks on a server with a GPU,
we typically prefer for the model's parameters to live on the GPU.

Next, we need to confirm that
the GPU version of MXNet is installed.
If a CPU version of MXNet is already installed,
we need to uninstall it first.
For example, use the `pip uninstall mxnet` command,
then install the corresponding MXNet version
according to your CUDA version.
Assuming you have CUDA 9.0 installed,
you can install the MXNet version
that supports CUDA 9.0 via `pip install mxnet-cu90`.
To run the programs in this section,
you need at least two GPUs.
:end_tab:

:begin_tab:`pytorch`
In PyTorch, every array has a device, we often refer it as a context.
So far, by default, all variables
and associated computation
have been assigned to the CPU.
Typically, other contexts might be various GPUs.
Things can get even hairier when
we deploy jobs across multiple servers.
By assigning arrays to contexts intelligently,
we can minimize the time spent
transferring data between devices.
For example, when training neural networks on a server with a GPU,
we typically prefer for the model's parameters to live on the GPU.

Next, we need to confirm that
the GPU version of PyTorch is installed.
If a CPU version of PyTorch is already installed,
we need to uninstall it first.
For example, use the `pip uninstall torch` command,
then install the corresponding MXNet version
according to your CUDA version.
Assuming you have CUDA 9.0 installed,
you can install the MXNet version
that supports CUDA 9.0 via `pip install torch-cu90`.
To run the programs in this section,
you need at least two GPUs.
:end_tab:


Note that this might be extravagant for most desktop computers
but it is easily available in the cloud, e.g.,
by using the AWS EC2 multi-GPU instances.
Almost all other sections do *not* require multiple GPUs.
Instead, this is simply to illustrate
how data flows between different devices.

## Computing Devices

We can specify devices, such as CPUs and GPUs,
for storage and calculation.
By default, tensors are created in the main memory
and then uses the CPU to calculate it.

:begin_tab:`mxnet`
In MXNet, the CPU and GPU can be indicated by `cpu()` and `gpu()`.
It should be noted that `cpu()`
(or any integer in the parentheses)
means all physical CPUs and memory.
This means that MXNet's calculations
will try to use all CPU cores.
However, `gpu()` only represents one card
and the corresponding memory.
If there are multiple GPUs, we use `gpu(i)`
to represent the $i^\mathrm{th}$ GPU ($i$ starts from 0).
Also, `gpu(0)` and `gpu()` are equivalent.
:end_tab:

:begin_tab:`pytorch`
In PyTorch, the CPU and GPU can be indicated by `torch.device('cpu')` and `torch.cuda.device('cuda')`.
It should be noted that the `cpu` device
means all physical CPUs and memory.
This means that PyTorch's calculations
will try to use all CPU cores.
However, a `gpu` device only represents one card
and the corresponding memory.
If there are multiple GPUs, we use `torch.cuda.device(f'cuda{i}')`
to represent the $i^\mathrm{th}$ GPU ($i$ starts from 0).
Also, `gpu:0` and `gpu` are equivalent.

:end_tab:

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

npx.cpu(), npx.gpu(), npx.gpu(1)
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn

torch.device('cpu'), torch.cuda.device('cuda'), torch.cuda.device('cuda:1')
```

We can query the number of available GPUs.

```{.python .input}
npx.num_gpus()
```

```{.python .input}
#@tab pytorch
torch.cuda.device_count()
```

Now we define two convenient functions that allow us
to run codes even if the requested GPUs do not exist.

```{.python .input}
def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    return npx.gpu(i) if npx.num_gpus() >= i + 1 else npx.cpu()

def try_all_gpus():  #@save
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    ctxes = [npx.gpu(i) for i in range(npx.num_gpus())]
    return ctxes if ctxes else [npx.cpu()]

try_gpu(), try_gpu(3), try_all_gpus()
```

```{.python .input}
#@tab pytorch
def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  #@save
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    ctxes = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return ctxes if ctxes else [torch.device('cpu')]

try_gpu(), try_gpu(3), try_all_gpus()
```

## Tensors and GPUs

By default, tensors are created on the CPU.
We can query the device where the tensor is located.

```{.python .input}
x = np.array([1, 2, 3])
x.ctx
```

```{.python .input}
#@tab pytorch
x = torch.tensor([1, 2, 3])
x.device
```

It is important to note that whenever we want
to operate on multiple terms,
they need to be in the same context.
For instance, if we sum two tensors,
we need to make sure that both arguments
live on the same device---otherwise the framework
would not know where to store the result
or even how to decide where to perform the computation.

### Storage on the GPU

There are several ways to store a tensor on the GPU.
For example, we can specify a storage device when creating a tensor.
Next, we create the tensor variable `a` on the first `gpu`.
Notice that when printing `a`, the device information changed.
The tensor created on a GPU only consumes the memory of this GPU.
We can use the `nvidia-smi` command to view GPU memory usage.
In general, we need to make sure we do not
create data that exceeds the GPU memory limit.

```{.python .input}
x = np.ones((2, 3), ctx=try_gpu())
x
```

```{.python .input}
#@tab pytorch
x = torch.ones(2, 3, device=try_gpu())
x
```

Assuming you have at least two GPUs, the following code will create a random array on the second GPU.

```{.python .input}
y = np.random.uniform(size=(2, 3), ctx=try_gpu(1))
y
```

```{.python .input}
#@tab pytorch
y = torch.randn(2, 3, device=try_gpu(1))
y
```

### Copying

If we want to compute $\mathbf{x} + \mathbf{y}$,
we need to decide where to perform this operation.
For instance, as shown in :numref:`fig_copyto`,
we can transfer $\mathbf{x}$ to the second GPU
and perform the operation there.
*Do not* simply add `x + y`,
since this will result in an exception.
The runtime engine would not know what to do,
it cannot find data on the same device and it fails.

![Copyto copies arrays to the target device](../img/copyto.svg)
:label:`fig_copyto`

`copyto` copies the data to another device such that we can add them.
Since $\mathbf{y}$ lives on the second GPU,
we need to move $\mathbf{x}$ there before we can add the two.

```{.python .input}
z = x.copyto(try_gpu(1))
print(x)
print(z)
```

```{.python .input}
#@tab pytorch
z = x.cuda(1)
print(x)
print(z)
```

Now that the data is on the same GPU
(both $\mathbf{z}$ and $\mathbf{y}$ are),
we can add them up.

```{.python .input}
y + z
```

```{.python .input}
#@tab pytorch
y + z
```

:begin_tab:`mxnet`

Imagine that your variable `z` already lives on your second GPU.
What happens if we call still `z.copyto(gpu(1))`?
It will make a copy and allocate new memory,
even though that variable already lives on the desired device!
There are times where, depending on the environment our code is running in,
two variables may already live on the same device.
So we only want to make a copy if the variables
currently lives on different contexts.
In these cases, we can call `as_in_ctx()`.
If the variable already live in the specified context
then this is a no-op.
Unless you specifically want to make a copy,
`as_in_ctx()` is the method of choice.
:end_tab:

:begin_tab:`pytorch`
Imagine that your variable `z` already lives on your second GPU.
What happens if we call still `z.cuda(1)`?
It will returns `z` instead of make a copy and allocate new memory.
:end_tab:

```{.python .input}
z.as_in_ctx(try_gpu(1)) is z
```

```{.python .input}
#@tab pytorch
z.cuda(1) is z
```

### Side Notes

People use GPUs to do machine learning
because they expect them to be fast.
But transferring variables between contexts is slow.
So we want you to be 100% certain
that you want to do something slow before we let you do it.
If the framework just did the copy automatically
without crashing then you might not realize
that you had written some slow code.

Also, transferring data between devices (CPU, GPUs, other machines)
is something that is *much slower* than computation.
It also makes parallelization a lot more difficult,
since we have to wait for data to be sent (or rather to be received)
before we can proceed with more operations.
This is why copy operations should be taken with great care.
As a rule of thumb, many small operations
are much worse than one big operation.
Moreover, several operations at a time
are much better than many single operations interspersed in the code
(unless you know what you are doing)
This is the case since such operations can block if one device
has to wait for the other before it can do something else.
It is a bit like ordering your coffee in a queue
rather than pre-ordering it by phone
and finding out that it is ready when you are.

Last, when we print tensors or convert tensors to the NumPy format,
if the data is not in main memory,
the framework will copy it to the main memory first,
resulting in additional transmission overhead.
Even worse, it is now subject to the dreaded Global Interpreter Lock
that makes everything wait for Python to complete.


## Neural Networks and GPUs

Similarly, a neural network model can specify devices.
The following code put the model parameters on the GPU
(we will see many more examples of
how to run models on GPUs in the following,
simply since they will become somewhat more compute intensive).

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(ctx=try_gpu())
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
```

When the input is a tensor on the GPU, Gluon will calculate the result on the same GPU.

```{.python .input}
net(x)
```

```{.python .input}
#@tab pytorch
net(x)
```

Let us confirm that the model parameters are stored on the same GPU.

```{.python .input}
net[0].weight.data().ctx
```

```{.python .input}
#@tab pytorch
net[0].weight.data.device
```

In short, as long as all data and parameters are on the same device, we can learn models efficiently. In the following we will see several such examples.

## Summary

* We can specify devices for storage and calculation, such as CPU or GPU.
  By default, data are created in the main memory
  and then uses the CPU for calculations.
* The framework requires all input data for calculation
  to be *on the same device*,
  be it CPU or the same GPU.
* You can lose significant performance by moving data without care.
  A typical mistake is as follows: computing the loss
  for every minibatch on the GPU and reporting it back
  to the user on the command line (or logging it in a NumPy array)
  will trigger a global interpreter lock which stalls all GPUs.
  It is much better to allocate memory
  for logging inside the GPU and only move larger logs.

## Exercises

1. Try a larger computation task, such as the multiplication of large matrices,
   and see the difference in speed between the CPU and GPU.
   What about a task with a small amount of calculations?
1. How should we read and write model parameters on the GPU?
1. Measure the time it takes to compute 1000
   matrix-matrix multiplications of $100 \times 100$ matrices
   and log the matrix norm $\mathrm{tr} M M^\top$ one result at a time
   vs. keeping a log on the GPU and transferring only the final result.
1. Measure how much time it takes to perform two matrix-matrix multiplications
   on two GPUs at the same time vs. in sequence
   on one GPU (hint: you should see almost linear scaling).

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/62)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/63)
:end_tab:
