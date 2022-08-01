```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Data Manipulation
:label:`sec_ndarray`

The photorealistic text-to-image examples in :numref:`fig_imagen` suggest that the T5 encoder alone may effectively represent text even without fine-tuning.

![Text-to-image examples by the Imagen model, whose text encoder is from T5 (figures taken from :citet:`saharia2022photorealistic`).](../img/imagen.png)
:width:`700px`
:label:`fig_imagen`




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
