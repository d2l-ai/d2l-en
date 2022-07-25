```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# GPUs
:label:`sec_use_gpu`

In :numref:`tab_intro_decade`, we discussed the rapid growth
of computation over the past two decades.
In a nutshell, GPU performance has increased
by a factor of 1000 every decade since 2000.
This offers great opportunities but it also suggests
a significant need to provide such performance.


In this section, we begin to discuss how to harness
this computational performance for your research.
First by using single GPUs and at a later point,
how to use multiple GPUs and multiple servers (with multiple GPUs).

Specifically, we will discuss how
to use a single NVIDIA GPU for calculations.
First, make sure you have at least one NVIDIA GPU installed.
Then, download the [NVIDIA driver and CUDA](https://developer.nvidia.com/cuda-downloads)
and follow the prompts to set the appropriate path.
Once these preparations are complete,
the `nvidia-smi` command can be used
to (**view the graphics card information**).

```{.python .input}
%%tab all
!nvidia-smi
```

:begin_tab:`mxnet`
You might have noticed that a MXNet tensor
looks almost identical to a NumPy `ndarray`.
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
Assuming you have CUDA 10.0 installed,
you can install the MXNet version
that supports CUDA 10.0 via `pip install mxnet-cu100`.
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
:end_tab:

To run the programs in this section,
you need at least two GPUs.
Note that this might be extravagant for most desktop computers
but it is easily available in the cloud, e.g.,
by using the AWS EC2 multi-GPU instances.
Almost all other sections do *not* require multiple GPUs.
Instead, this is simply to illustrate
how data flow between different devices.

## [**Computing Devices**]

We can specify devices, such as CPUs and GPUs,
for storage and calculation.
By default, tensors are created in the main memory
and then use the CPU to calculate it.

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
In PyTorch, the CPU and GPU can be indicated by `torch.device('cpu')` and `torch.device('cuda')`.
It should be noted that the `cpu` device
means all physical CPUs and memory.
This means that PyTorch's calculations
will try to use all CPU cores.
However, a `gpu` device only represents one card
and the corresponding memory.
If there are multiple GPUs, we use `torch.device(f'cuda:{i}')`
to represent the $i^\mathrm{th}$ GPU ($i$ starts from 0).
Also, `gpu:0` and `gpu` are equivalent.
:end_tab:

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab all
def cpu():  #@save
    if tab.selected('mxnet'):
        return npx.cpu()
    if tab.selected('pytorch'):
        return torch.device('cpu')
    if tab.selected('tensorflow'):
        return tf.device('/CPU:0')

def gpu(i=0):  #@save
    if tab.selected('mxnet'):
        return npx.gpu(i)
    if tab.selected('pytorch'):
        return torch.device(f'cuda:{i}')
    if tab.selected('tensorflow'):
        return tf.device(f'/GPU:{i}')

cpu(), gpu(), gpu(1)
```

We can (**query the number of available GPUs.**)

```{.python .input}
%%tab all
def num_gpus():  #@save
    if tab.selected('mxnet'):
        return npx.num_gpus()
    if tab.selected('pytorch'):
        return torch.cuda.device_count()
    if tab.selected('tensorflow'):
        return len(tf.config.experimental.list_physical_devices('GPU'))

num_gpus()
```

Now we [**define two convenient functions that allow us
to run code even if the requested GPUs do not exist.**]

```{.python .input}
%%tab all
def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if num_gpus() >= i + 1:
        return gpu(i)
    return cpu()

def try_all_gpus():  #@save
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    return [gpu(i) for i in range(num_gpus())]

try_gpu(), try_gpu(10), try_all_gpus()
```

## Tensors and GPUs

By default, tensors are created on the CPU.
We can [**query the device where the tensor is located.**]

```{.python .input}
%%tab mxnet
x = np.array([1, 2, 3])
x.ctx
```

```{.python .input}
%%tab pytorch
x = torch.tensor([1, 2, 3])
x.device
```

```{.python .input}
%%tab tensorflow
x = tf.constant([1, 2, 3])
x.device
```

It is important to note that whenever we want
to operate on multiple terms,
they need to be on the same device.
For instance, if we sum two tensors,
we need to make sure that both arguments
live on the same device---otherwise the framework
would not know where to store the result
or even how to decide where to perform the computation.

### Storage on the GPU

There are several ways to [**store a tensor on the GPU.**]
For example, we can specify a storage device when creating a tensor.
Next, we create the tensor variable `X` on the first `gpu`.
The tensor created on a GPU only consumes the memory of this GPU.
We can use the `nvidia-smi` command to view GPU memory usage.
In general, we need to make sure that we do not create data that exceeds the GPU memory limit.

```{.python .input}
%%tab mxnet
X = np.ones((2, 3), ctx=try_gpu())
X
```

```{.python .input}
%%tab pytorch
X = torch.ones(2, 3, device=try_gpu())
X
```

```{.python .input}
%%tab tensorflow
with try_gpu():
    X = tf.ones((2, 3))
X
```

Assuming that you have at least two GPUs, the following code will (**create a random tensor on the second GPU.**)

```{.python .input}
%%tab mxnet
Y = np.random.uniform(size=(2, 3), ctx=try_gpu(1))
Y
```

```{.python .input}
%%tab pytorch
Y = torch.rand(2, 3, device=try_gpu(1))
Y
```

```{.python .input}
%%tab tensorflow
with try_gpu(1):
    Y = tf.random.uniform((2, 3))
Y
```

### Copying

[**If we want to compute `X + Y`,
we need to decide where to perform this operation.**]
For instance, as shown in :numref:`fig_copyto`,
we can transfer `X` to the second GPU
and perform the operation there.
*Do not* simply add `X` and `Y`,
since this will result in an exception.
The runtime engine would not know what to do:
it cannot find data on the same device and it fails.
Since `Y` lives on the second GPU,
we need to move `X` there before we can add the two.

![Copy data to perform an operation on the same device.](../img/copyto.svg)
:label:`fig_copyto`

```{.python .input}
%%tab mxnet
Z = X.copyto(try_gpu(1))
print(X)
print(Z)
```

```{.python .input}
%%tab pytorch
Z = X.cuda(1)
print(X)
print(Z)
```

```{.python .input}
%%tab tensorflow
with try_gpu(1):
    Z = X
print(X)
print(Z)
```

Now that [**the data is on the same GPU
(both `Z` and `Y` are),
we can add them up.**]

```{.python .input}
%%tab all
Y + Z
```

:begin_tab:`mxnet`
Imagine that your variable `Z` already lives on your second GPU.
What happens if we still call  `Z.copyto(gpu(1))`?
It will make a copy and allocate new memory,
even though that variable already lives on the desired device.
There are times where, depending on the environment our code is running in,
two variables may already live on the same device.
So we want to make a copy only if the variables
currently live in different devices.
In these cases, we can call `as_in_ctx`.
If the variable already live in the specified device
then this is a no-op.
Unless you specifically want to make a copy,
`as_in_ctx` is the method of choice.
:end_tab:

:begin_tab:`pytorch`
Imagine that your variable `Z` already lives on your second GPU.
What happens if we still call `Z.cuda(1)`?
It will return `Z` instead of making a copy and allocating new memory.
:end_tab:

:begin_tab:`tensorflow`
Imagine that your variable `Z` already lives on your second GPU.
What happens if we still call `Z2 = Z` under the same device scope?
It will return `Z` instead of making a copy and allocating new memory.
:end_tab:

```{.python .input}
%%tab mxnet
Z.as_in_ctx(try_gpu(1)) is Z
```

```{.python .input}
%%tab pytorch
Z.cuda(1) is Z
```

```{.python .input}
%%tab tensorflow
with try_gpu(1):
    Z2 = Z
Z2 is Z
```

### Side Notes

People use GPUs to do machine learning
because they expect them to be fast.
But transferring variables between devices is slow.
So we want you to be 100% certain
that you want to do something slow before we let you do it.
If the deep learning framework just did the copy automatically
without crashing then you might not realize
that you had written some slow code.

Also, transferring data between devices (CPU, GPUs, and other machines)
is something that is much slower than computation.
It also makes parallelization a lot more difficult,
since we have to wait for data to be sent (or rather to be received)
before we can proceed with more operations.
This is why copy operations should be taken with great care.
As a rule of thumb, many small operations
are much worse than one big operation.
Moreover, several operations at a time
are much better than many single operations interspersed in the code
unless you know what you are doing.
This is the case since such operations can block if one device
has to wait for the other before it can do something else.
It is a bit like ordering your coffee in a queue
rather than pre-ordering it by phone
and finding out that it is ready when you are.

Last, when we print tensors or convert tensors to the NumPy format,
if the data is not in the main memory,
the framework will copy it to the main memory first,
resulting in additional transmission overhead.
Even worse, it is now subject to the dreaded global interpreter lock
that makes everything wait for Python to complete.


## [**Neural Networks and GPUs**]

Similarly, a neural network model can specify devices.
The following code puts the model parameters on the GPU.

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(ctx=try_gpu())
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(1))
net = net.to(device=try_gpu())
```

```{.python .input}
%%tab tensorflow
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    net = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1)])
```

We will see many more examples of
how to run models on GPUs in the following chapters,
simply since they will become somewhat more computationally intensive.

When the input is a tensor on the GPU, the model will calculate the result on the same GPU.

```{.python .input}
%%tab all
net(X)
```

Let's (**confirm that the model parameters are stored on the same GPU.**)

```{.python .input}
%%tab mxnet
net[0].weight.data().ctx
```

```{.python .input}
%%tab pytorch
net[0].weight.data.device
```

```{.python .input}
%%tab tensorflow
net.layers[0].weights[0].device, net.layers[0].weights[1].device
```

Let the trainer support GPU.

```{.python .input}
%%tab mxnet
@d2l.add_to_class(d2l.Module)  #@save
def set_scratch_params_device(self, device):
    for attr in dir(self):
        a = getattr(self, attr)
        if isinstance(a, np.ndarray):
            with autograd.record():
                setattr(self, attr, a.as_in_ctx(device))
            getattr(self, attr).attach_grad()
        if isinstance(a, d2l.Module):
            a.set_scratch_params_device(device)
        if isinstance(a, list):
            for elem in a:
                elem.set_scratch_params_device(device)
```

```{.python .input}
%%tab mxnet, pytorch
@d2l.add_to_class(d2l.Trainer)  #@save
def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
    self.save_hyperparameters()
    self.gpus = [d2l.gpu(i) for i in range(min(num_gpus, d2l.num_gpus()))]

@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_batch(self, batch):
    if self.gpus:
        batch = [d2l.to(a, self.gpus[0]) for a in batch]
    return batch

@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_model(self, model):
    model.trainer = self
    model.board.xlim = [0, self.max_epochs]
    if self.gpus:
        if tab.selected('mxnet'):
            model.collect_params().reset_ctx(self.gpus[0])
            model.set_scratch_params_device(self.gpus[0])
        if tab.selected('pytorch'):
            model.to(self.gpus[0])
    self.model = model
```

In short, as long as all data and parameters are on the same device, we can learn models efficiently. In the following chapters we will see several such examples.

## Summary

* We can specify devices for storage and calculation, such as the CPU or GPU.
  By default, data is created in the main memory
  and then uses the CPU for calculations.
* The deep learning framework requires all input data for calculation
  to be on the same device,
  be it CPU or the same GPU.
* You can lose significant performance by moving data without care.
  A typical mistake is as follows: computing the loss
  for every minibatch on the GPU and reporting it back
  to the user on the command line (or logging it in a NumPy `ndarray`)
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
   and log the Frobenius norm of the output matrix one result at a time
   vs. keeping a log on the GPU and transferring only the final result.
1. Measure how much time it takes to perform two matrix-matrix multiplications
   on two GPUs at the same time vs. in sequence
   on one GPU. Hint: you should see almost linear scaling.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/62)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/63)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/270)
:end_tab:


