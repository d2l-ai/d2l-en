```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Data Manipulation
:label:`sec_ndarray`

In order to get anything done,
we need some way to store and manipulate data.
Generally, there are two important things
we need to do with data:
(i) acquire them;
and (ii) process them once they are inside the computer.
There is no point in acquiring data
without some way to store it,
so to start, let's get our hands dirty
with $n$-dimensional arrays,
which we also call *tensors*.
If you already know the NumPy
scientific computing package,
this will be a breeze.
For all modern deep learning frameworks,
the *tensor class* (`ndarray` in MXNet,
`Tensor` in PyTorch and TensorFlow)
resembles NumPy's `ndarray`,
with a few killer features added.
First, the tensor class
supports automatic differentiation.
Second, it leverages GPUs
to accelerate numerical computation,
whereas NumPy only runs on CPUs.
These properties make neural networks
both easy to code and fast to run.



## Getting Started

:begin_tab:`mxnet`
To start, we import the `np` (`numpy`) and
`npx` (`numpy_extension`) modules from MXNet.
Here, the `np` module includes
functions supported by NumPy,
while the `npx` module contains a set of extensions
developed to empower deep learning
within a NumPy-like environment.
When using tensors, we almost always
invoke the `set_np` function:
this is for compatibility of tensor processing
by other components of MXNet.
:end_tab:

:begin_tab:`pytorch`
(**To start, we import the PyTorch library.
Note that the package name is `torch`.**)
:end_tab:

:begin_tab:`tensorflow`
To start, we import `tensorflow`.
For brevity, practitioners
often assign the alias `tf`.
:end_tab:

```{.python .input}
%%tab mxnet
1 + 1
```

```{.python .input}
%%tab pytorch
1 + 1
```

```{.python .input}
%%tab tensorflow
1 + 1
```

[**A tensor represents a (possibly multi-dimensional) array of numerical values.**]
With one axis, a tensor is called a *vector*.
With two axes, a tensor is called a *matrix*.
With $k > 2$ axes, we drop the specialized names
and just refer to the object as a $k^\mathrm{th}$ *order tensor*.
