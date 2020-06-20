# Data Manipulation
:label:`sec_ndarray`

In order to get anything done, we need some way to store and manipulate data.
Generally, there are two important things we need to do with data: (i) acquire
them; and (ii) process them once they are inside the computer.  There is no
point in acquiring data without some way to store it, so let us get our hands
dirty first by playing with synthetic data.  To start, we introduce the
$n$-dimensional array, which is also called the *tensor*.

If you have worked with NumPy, the most widely-used
scientific computing package in Python,
then you will find this section familiar.
No matter which framework you use,
its *tensor class* (`ndarray` in MXNet,
`Tensor` in both PyTorch and TensorFlow) is similar to NumPy's `ndarray` with
a few killer features.
First, GPU is well-supported to accelerate the computation
whereas NumPy only supports CPU computation.
Second, the tensor class
supports automatic differentiation.
These properties make the tensor class suitable for deep learning.
Throughout the book, when we say tensors,
we are referring to instances of the tensor class unless otherwise stated.

## Getting Started

In this section, we aim to get you up and running,
equipping you with the basic math and numerical computing tools
that you will build on as you progress through the book.
Do not worry if you struggle to grok some of
the mathematical concepts or library functions.
The following sections will revisit this material
in the context of practical examples and it will sink.
On the other hand, if you already have some background
and want to go deeper into the mathematical content, just skip this section.

:begin_tab:`mxnet`
To start, we import the `np` (`numpy`) and
`npx` (`numpy_extension`) modules from MXNet.
Here, the `np` module includes functions supported by NumPy,
while the `npx` module contains a set of extensions
developed to empower deep learning within a NumPy-like environment.
When using tensors, we almost always invoke the `set_np` function:
this is for compatibility of tensor processing by other components of MXNet.
:end_tab:

:begin_tab:`pytorch`
To start, we import `torch`. Note that though it's called PyTorch, we should
import `torch` instead of `pytorch`.
:end_tab:

:begin_tab:`tensorflow`
To start, we import `tensorflow`. As the name is a little long, we often import
it with a short alias `tf`.
:end_tab:

:begin_tab:`jax`
To start, we import `jax.numpy as np`. That's right! Instead of importing `numpy`, we import `numpy` from `JAX`! This is because `JAX` aims to mirror the `numpy` interface while providing additional functionality for numeric computation work.
:end_tab:

```{.python .input}
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
import torch
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf
```

```{.python .input}
#@tab jax
import jax
import jax.numpy as np
```

A tensor represents a (possibly multi-dimensional) array of numerical values.
With one axis, a tensor corresponds (in math) to a *vector*.
With two axes, a tensor corresponds to a *matrix*.
Tensors with more than two axes do not have special
mathematical names.

To start, we can use `arange` to create a row vector `x`
containing the first 12 integers starting with 0,
though they are created as floats by default.
Each of the values in a tensor is called an *element* of the tensor.
For instance, there are 12 elements in the tensor `x`.
Unless otherwise specified, a new tensor
will be stored in main memory and designated for CPU-based computation.

```{.python .input}
x = np.arange(12)
x
```

```{.python .input}
#@tab pytorch
x = torch.arange(12)
x
```

```{.python .input}
#@tab tensorflow
x = tf.constant(range(12))
x
```

```{.python .input}
#@tab jax
x = np.arange(12)
x
```

We can access a tensor's *shape* (the length along each axis)
by inspecting its `shape` property.

```{.python .input}
x.shape
```

```{.python .input}
#@tab pytorch
x.shape
```

```{.python .input}
#@tab tensorflow
x.shape
```

```{.python .input}
#@tab jax
x.shape
```

If we just want to know the total number of elements in a tensor,
i.e., the product of all of the shape elements,
we can inspect its `size` property..
Because we are dealing with a vector here,
the single element of its `shape` is identical to its `size`.

```{.python .input}
x.size
```

```{.python .input}
#@tab pytorch
x.size()
```

```{.python .input}
#@tab tensorflow
tf.size(x)
```

```{.python .input}
#@tab jax
x.size
```

To change the shape of a tensor without altering
either the number of elements or their values,
we can invoke the `reshape` function.
For example, we can transform our tensor, `x`,
from a row vector with shape (12,) to a matrix with shape (3, 4).
This new tensor contains the exact same values,
but views them as a matrix organized as 3 rows and 4 columns.
To reiterate, although the shape has changed,
the elements in `x` have not.
Note that the `size` is unaltered by reshaping.

```{.python .input}
x = x.reshape(3, 4)
x
```

```{.python .input}
#@tab pytorch
x = x.reshape((3, 4))
x
```

```{.python .input}
#@tab tensorflow
x = tf.reshape(x, (3, 4))
x
```

```{.python .input}
#@tab jax
x = x.reshape(3, 4)
x
```

Reshaping by manually specifying every dimension is unnecessary.
If our target shape is a matrix with shape (height, width),
then after we know the width, the height is given implicitly.
Why should we have to perform the division ourselves?
In the example above, to get a matrix with 3 rows,
we specified both that it should have 3 rows and 4 columns.
Fortunately, tensors can automatically work out one dimension given the rest.
We invoke this capability by placing `-1` for the dimension
that we would like tensors to automatically infer.
In our case, instead of calling `x.reshape(3, 4)`,
we could have equivalently called `x.reshape(-1, 4)` or `x.reshape(3, -1)`.

Typically, we will want our matrices initialized
either with zeros, ones, some other constants,
or numbers randomly sampled from a specific distribution.
We can create a tensor representing a tensor with all elements
set to 0 and a shape of (2, 3, 4) as follows:

```{.python .input}
np.zeros((2, 3, 4))
```

```{.python .input}
#@tab pytorch
torch.zeros(2, 3, 4)
```

```{.python .input}
#@tab tensorflow
tf.zeros((2, 3, 4))
```

```{.python .input}
#@tab jax
np.zeros((2, 3, 4))
```

Similarly, we can create tensors with each element set to 1 as follows:

```{.python .input}
np.ones((2, 3, 4))
```

```{.python .input}
#@tab pytorch
torch.ones((2, 3, 4))
```

```{.python .input}
#@tab tensorflow
tf.ones((2, 3, 4))
```

```{.python .input}
#@tab jax
np.ones((2, 3, 4))
```

Often, we want to randomly sample the values
for each element in a tensor
from some probability distribution.
For example, when we construct arrays to serve
as parameters in a neural network, we will
typically initialize their values randomly.
The following snippet creates a tensor with shape (3, 4).
Each of its elements is randomly sampled
from a standard Gaussian (normal) distribution
with a mean of 0 and a standard deviation of 1.

:begin_tab:`jax`
Randomness works slightly differently in JAX. Every call to `jax.random` needs to include a key, which we must generate beforehand. Once that's done, the process is very similar to `numpy`.
:end_tab:

```{.python .input}
np.random.normal(0, 1, size=(3, 4))
```

```{.python .input}
#@tab pytorch
torch.randn(3, 4)
```

```{.python .input}
#@tab tensorflow
tf.random.normal(shape=[3, 4])
```

```{.python .input}
#@tab jax
key = jax.random.PRNGKey(seed=42)
jax.random.normal(key, shape=(3,4))
```

We can also specify the exact values for each element in the desired tensor
by supplying a Python list (or list of lists) containing the numerical values.
Here, the outermost list corresponds to axis 0, and the inner list to axis 1.

```{.python .input}
np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
#@tab pytorch
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
#@tab tensorflow
tf.constant([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
#@tab jax
np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

## Operations

This book is not about software engineering.
Our interests are not limited to simply
reading and writing data from/to arrays.
We want to perform mathematical operations on those arrays.
Some of the simplest and most useful operations
are the *elementwise* operations.
These apply a standard scalar operation
to each element of an array.
For functions that take two arrays as inputs,
elementwise operations apply some standard binary operator
on each pair of corresponding elements from the two arrays.
We can create an elementwise function from any function
that maps from a scalar to a scalar.

In mathematical notation, we would denote such
a *unary* scalar operator (taking one input)
by the signature $f: \mathbb{R} \rightarrow \mathbb{R}$.
This just means that the function is mapping
from any real number ($\mathbb{R}$) onto another.
Likewise, we denote a *binary* scalar operator
(taking two real inputs, and yielding one output)
by the signature $f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$.
Given any two vectors $\mathbf{u}$ and $\mathbf{v}$ *of the same shape*,
and a binary operator $f$, we can produce a vector
$\mathbf{c} = F(\mathbf{u},\mathbf{v})$
by setting $c_i \gets f(u_i, v_i)$ for all $i$,
where $c_i, u_i$, and $v_i$ are the $i^\mathrm{th}$ elements
of vectors $\mathbf{c}, \mathbf{u}$, and $\mathbf{v}$.
Here, we produced the vector-valued
$F: \mathbb{R}^d, \mathbb{R}^d \rightarrow \mathbb{R}^d$
by *lifting* the scalar function to an elementwise vector operation.

The common standard arithmetic operators
(`+`, `-`, `*`, `/`, and `**`)
have all been *lifted* to elementwise operations
for any identically-shaped tensors of arbitrary shape.
We can call elementwise operations on any two tensors of the same shape.
In the following example, we use commas to formulate a 5-element tuple,
where each element is the result of an elementwise operation.

```{.python .input}
x = np.array([1, 2, 4, 8])
y = np.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # The ** operator is exponentiation
```

```{.python .input}
#@tab pytorch
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # The ** operator is exponentiation
```

```{.python .input}
#@tab tensorflow
x = tf.constant([1.0, 2, 4, 8])
y = tf.constant([2.0, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # The ** operator is exponentiation
```

```{.python .input}
#@tab jax
x = np.array([1, 2, 4, 8])
y = np.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # The ** operator is exponentiation
```

Many more operations can be applied elementwise,
including unary operators like exponentiation.

```{.python .input}
np.exp(x)
```

```{.python .input}
#@tab pytorch
torch.exp(x)
```

```{.python .input}
#@tab tensorflow
tf.exp(x)
```

```{.python .input}
#@tab jax
np.exp(x)
```

In addition to elementwise computations,
we can also perform linear algebra operations,
including vector dot products and matrix multiplication.
We will explain the crucial bits of linear algebra
(with no assumed prior knowledge) in :numref:`sec_linear-algebra`.

We can also *concatenate* multiple tensors together,
stacking them end-to-end to form a larger tensor.
We just need to provide a list of tensors
and tell the system along which axis to concatenate.
The example below shows what happens when we concatenate
two matrices along rows (axis 0, the first element of the shape)
vs. columns (axis 1, the second element of the shape).
We can see that the first output tensor's axis-0 length ($6$)
is the sum of the two input tensors' axis-0 lengths ($3 + 3$);
while the second output tensor's axis-1 length ($8$)
is the sum of the two input tensors' axis-1 lengths ($4 + 4$).

```{.python .input}
x = np.arange(12).reshape(3, 4)
y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
np.concatenate([x, y], axis=0), np.concatenate([x, y], axis=1)
```

```{.python .input}
#@tab pytorch
x = torch.arange(12, dtype=torch.float32).reshape((3,4))
y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((x, y), dim=0), torch.cat((x, y), dim=1)
```

```{.python .input}
#@tab tensorflow
x = tf.constant(range(12), dtype=tf.float32, shape=(3, 4))
y = tf.constant([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
tf.concat([x, y], axis=0), tf.concat([x, y], axis=1)
```

```{.python .input}
#@tab jax
x = np.arange(12).reshape(3, 4)
y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
np.concatenate([x, y], axis=0), np.concatenate([x, y], axis=1)
```

Sometimes, we want to construct a binary tensor via *logical statements*.
Take `x == y` as an example.
For each position, if `x` and `y` are equal at that position,
the corresponding entry in the new tensor takes a value of 1,
meaning that the logical statement `x == y` is true at that position;
otherwise that position takes 0.

```{.python .input}
x == y
```

```{.python .input}
#@tab pytorch
x == y
```

```{.python .input}
#@tab tensorflow
x == y
```

```{.python .input}
#@tab jax
x == y
```

Summing all the elements in the tensor yields a tensor with only one element.

```{.python .input}
x.sum()
```

```{.python .input}
#@tab pytorch
x.sum()
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(x)
```

```{.python .input}
#@tab jax
x.sum()
```

## Broadcasting Mechanism
:label:`subsec_broadcasting`

In the above section, we saw how to perform elementwise operations
on two tensors of the same shape. Under certain conditions,
even when shapes differ, we can still perform elementwise operations
by invoking the *broadcasting mechanism*.
This mechanism works in the following way:
First, expand one or both arrays
by copying elements appropriately
so that after this transformation,
the two tensors have the same shape.
Second, carry out the elementwise operations
on the resulting arrays.

In most cases, we broadcast along an axis where an array
initially only has length 1, such as in the following example:

```{.python .input}
a = np.arange(3).reshape(3, 1)
b = np.arange(2).reshape(1, 2)
a, b
```

```{.python .input}
#@tab pytorch
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
```

```{.python .input}
#@tab tensorflow
a = tf.constant(range(3), shape=(3, 1))
b = tf.constant(range(2), shape=(1, 2))
a, b
```

```{.python .input}
#@tab jax
a = np.arange(3).reshape(3, 1)
b = np.arange(2).reshape(1, 2)
a, b
```

Since `a` and `b` are $3\times1$ and $1\times2$ matrices respectively,
their shapes do not match up if we want to add them.
We *broadcast* the entries of both matrices into a larger $3\times2$ matrix as follows:
for matrix `a` it replicates the columns
and for matrix `b` it replicates the rows
before adding up both elementwise.

```{.python .input}
a + b
```

```{.python .input}
#@tab pytorch
a + b
```

```{.python .input}
#@tab tensorflow
a + b
```

```{.python .input}
#@tab jax
a + b
```

## Indexing and Slicing

Just as in any other Python array, elements in a tensor can be accessed by index.
As in any Python array, the first element has index 0
and ranges are specified to include the first but *before* the last element.
As in standard Python lists, we can access elements
according to their relative position to the end of the list
by using negative indices.

Thus, `[-1]` selects the last element and `[1:3]`
selects the second and the third elements as follows:

```{.python .input}
x[-1], x[1:3]
```

```{.python .input}
#@tab pytorch
x[-1], x[1:3]
```

```{.python .input}
#@tab tensorflow
x[-1], x[1:3]
```

```{.python .input}
#@tab jax
x[-1], x[1:3]
```

:begin_tab:`jax`
In regular `numpy`, we can write elements of a matrix by specifying indices. In `JAX` this is not permitted due to the functional paradigm followed by `JAX`, due to which arrays are immutable. However, it is still possible to perform a similar operation as follows:
:end_tab:

```{.python .input}
x[1, 2] = 9
x
```

```{.python .input}
#@tab pytorch
x[1, 2] = 9
x
```

```{.python .input}
#@tab tensorflow
x = tf.convert_to_tensor(tf.Variable(x)[1, 2].assign(9))
x
```

```{.python .input}
#@tab jax
#we want to achieve x[1, 2] = 9
x_new = jax.ops.index_update(x, jax.ops.index[1,2], 9)

x_new
```

If we want to assign multiple elements the same value,
we simply index all of them and then assign them the value.
For instance, `[0:2, :]` accesses the first and second rows,
where `:` takes all the elements along axis 1 (column).
While we discussed indexing for matrices,
this obviously also works for vectors
and for tensors of more than 2 dimensions.

```{.python .input}
x[0:2, :] = 12
x
```

```{.python .input}
#@tab pytorch
x[0:2, :] = 12
x
```

```{.python .input}
#@tab tensorflow
x_var = tf.Variable(x)
x_var[1:2,:].assign(tf.ones(x_var[1:2,:].shape, dtype = tf.float32)*12)
x = tf.convert_to_tensor(x_var)
x
```

```{.python .input}
#@tab jax
x_new = jax.ops.index_update(x, jax.ops.index[0:2, :], 12)
x_new
```

## Saving Memory

Running operations can cause new memory to be
allocated to host results.
For example, if we write `y = x + y`,
we will dereference the tensor that `y` used to point to
and instead point `y` at the newly allocated memory.
In the following example, we demonstrate this with Python's `id()` function,
which gives us the exact address of the referenced object in memory.
After running `y = y + x`, we will find that `id(y)` points to a different location.
That is because Python first evaluates `y + x`,
allocating new memory for the result and then makes `y`
point to this new location in memory.

```{.python .input}
before = id(y)
y = y + x
id(y) == before
```

```{.python .input}
#@tab pytorch
before = id(y)
y = y + x
id(y) == before
```

```{.python .input}
#@tab tensorflow
before = id(y)
y = y + x
id(y) == before
```

```{.python .input}
#@tab jax
before = id(y)
y = y + x
id(y) == before #we have overwritten y!
```

This might be undesirable for two reasons.
First, we do not want to run around
allocating memory unnecessarily all the time.
In machine learning, we might have
hundreds of megabytes of parameters
and update all of them multiple times per second.
Typically, we will want to perform these updates *in place*.
Second, we might point at the same parameters from multiple variables.
If we do not update in place, other references will still point to
the old memory location, making it possible for parts of our code
to inadvertently reference stale parameters.

:begin_tab:`jax`
Although in other frameworks, inplace updating is common, it comes with certain risks and thus all arrays in JAX are immutable and cannot be updated in place. If we attempt to perform an in-place update, we will always get a new copy of an object at a different memory location, although it may seem like it is the same object.
:end_tab:

```{.python .input}
z = np.zeros_like(y)
print('id(z):', id(z))
z[:] = x + y
print('id(z):', id(z))
```

```{.python .input}
#@tab pytorch
z = torch.zeros_like(y)
print('id(z):', id(z))
z[:] = x + y
print('id(z):', id(z))
```

```{.python .input}
#@tab tensorflow
z = tf.Variable(tf.zeros_like(y))
print('id(z):', id(z))
z[:].assign(x + y)
print('id(z):', id(z))
```

```{.python .input}
#@tab jax
z = np.zeros_like(y)
print('id(z):', id(z))
z = jax.ops.index_update(z, jax.ops.index[:], x+y)
print('id(z):', id(z))
```

:begin_tab:`jax`
The same holds for in-place addition, which is possible in `numpy` and other frameworks but would result in an error in `JAX`. In the example below, `x` changes location in memory and the original object loses its reference.
:end_tab:

```{.python .input}
before = id(x)
x += y
id(x) == before
```

```{.python .input}
#@tab pytorch
before = id(x)
x += y
id(x) == before
```

```{.python .input}
#@tab tensorflow
before = id(x)
tf.Variable(x).assign(x + y)
id(x) == before
```

```{.python .input}
#@tab jax
before = id(x)
x = jax.ops.index_add(x, jax.ops.index[:], y)
id(x) == before
```

## Conversion to Other Python Objects

:begin_tab:`jax`
Converting to a NumPy tensor, or vice versa, requires importing the original `numpy`. As always, the converted result does not share memory.
This minor inconvenience is actually quite important:
when you perform operations on the CPU or on GPUs,
you do not want JAX to halt computation, waiting to see
whether the NumPy package of Python might want to be doing something else
with the same chunk of memory.
:end_tab:

```{.python .input}
a = x.asnumpy()
b = np.array(a)
type(a), type(b)
```

```{.python .input}
#@tab pytorch
a = x.numpy()
b = torch.tensor(a)
type(a), type(b)
```

```{.python .input}
#@tab tensorflow
a = x.numpy()
b = tf.constant(a)
type(a), type(b)
```

```{.python .input}
#@tab jax
import numpy as onp # 'original numpy'
a = onp.array(x)
b = np.array(a)
type(a), type(b)
```

To convert a size-1 tensor to a Python scalar,
we can invoke the `item` function or Python's built-in functions.

```{.python .input}
a = np.array([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
#@tab pytorch
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
#@tab tensorflow
a = tf.constant([3.5]).numpy()
a, a.item(), float(a), int(a)
```

```{.python .input}
#@tab jax
a = np.array([3.5])
a, a.item(), float(a), int(a)
```

## Summary

* The main interface to store and manipulate data for deep learning is the tensor ($n$-dimensional array). It provides a variety of functionalities including basic mathematics operations, broadcasting, indexing, slicing, memory saving, and conversion to other Python objects.


## Exercises

1. Run the code in this section. Change the conditional statement `x == y` in this section to `x < y` or `x > y`, and then see what kind of tensor you can get.
1. Replace the two tensors that operate by element in the broadcasting mechanism with other shapes, e.g., 3-dimensional tensors. Is the result the same as expected?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/26)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/27)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/187)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/187)
:end_tab:
