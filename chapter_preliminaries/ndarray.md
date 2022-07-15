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
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
import torch
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
```

[**A tensor represents a (possibly multi-dimensional) array of numerical values.**]
With one axis, a tensor is called a *vector*.
With two axes, a tensor is called a *matrix*.
With $k > 2$ axes, we drop the specialized names
and just refer to the object as a $k^\mathrm{th}$ *order tensor*.

:begin_tab:`mxnet`
MXNet provides a variety of functions 
for creating new tensors 
prepopulated with values. 
For example, by invoking `arange(n)`,
we can create a vector of evenly spaced values,
starting at 0 (included) 
and ending at `n` (not included).
By default, the interval size is $1$.
Unless otherwise specified, 
new tensors are stored in main memory 
and designated for CPU-based computation.
:end_tab:

:begin_tab:`pytorch`
PyTorch provides a variety of functions 
for creating new tensors 
prepopulated with values. 
For example, by invoking `arange(n)`,
we can create a vector of evenly spaced values,
starting at 0 (included) 
and ending at `n` (not included).
By default, the interval size is $1$.
Unless otherwise specified, 
new tensors are stored in main memory 
and designated for CPU-based computation.
:end_tab:

:begin_tab:`tensorflow`
TensorFlow provides a variety of functions 
for creating new tensors 
prepopulated with values. 
For example, by invoking `range(n)`,
we can create a vector of evenly spaced values,
starting at 0 (included) 
and ending at `n` (not included).
By default, the interval size is $1$.
Unless otherwise specified, 
new tensors are stored in main memory 
and designated for CPU-based computation.
:end_tab:

```{.python .input}
%%tab mxnet
x = np.arange(12)
x
```

```{.python .input}
%%tab pytorch
x = torch.arange(12, dtype=torch.float32)
x
```

```{.python .input}
%%tab tensorflow
x = tf.range(12, dtype=tf.float32)
x
```

:begin_tab:`mxnet`
Each of these values is called
an *element* of the tensor.
The tensor `x` contains 12 elements.
We can inspect the total number of elements 
in a tensor via its `size` attribute.
:end_tab:

:begin_tab:`pytorch`
Each of these values is called
an *element* of the tensor.
The tensor `x` contains 12 elements.
We can inspect the total number of elements 
in a tensor via its `numel` method.
:end_tab:

:begin_tab:`tensorflow`
Each of these values is called
an *element* of the tensor.
The tensor `x` contains 12 elements.
We can inspect the total number of elements 
in a tensor via the `size` function.
:end_tab:

```{.python .input}
%%tab mxnet
x.size
```

```{.python .input}
%%tab pytorch
x.numel()
```

```{.python .input}
%%tab tensorflow
tf.size(x)
```

(**We can access a tensor's *shape***) 
(the length along each axis)
by inspecting its `shape` attribute.
Because we are dealing with a vector here,
the `shape` contains just a single element
and is identical to the size.

```{.python .input}
%%tab all
x.shape
```

We can [**change the shape of a tensor
without altering its size or values**],
by invoking `reshape`.
For example, we can transform 
our vector `x` whose shape is (12,) 
to a matrix `X`  with shape (3, 4).
This new tensor retains all elements
but reconfigures them into a matrix.
Notice that the elements of our vector
are laid out one row at a time and thus
`x[3] == X[0, 3]`.

```{.python .input}
%%tab mxnet, pytorch
X = x.reshape(3, 4)
X
```

```{.python .input}
%%tab tensorflow
X = tf.reshape(x, (3, 4))
X
```

Note that specifying every shape component
to `reshape` is redundant.
Because we already know our tensor's size,
we can work out one component of the shape given the rest.
For example, given a tensor of size $n$
and target shape ($h$, $w$),
we know that $w = n/h$.
To automatically infer one component of the shape,
we can place a `-1` for the shape component
that should be inferred automatically.
In our case, instead of calling `x.reshape(3, 4)`,
we could have equivalently called `x.reshape(-1, 4)` or `x.reshape(3, -1)`.

Practitioners often need to work with tensors
initialized to contain all zeros or ones.
[**We can construct a tensor with all elements set to zero**] (~~or one~~)
and a shape of (2, 3, 4) via the `zeros` function.

```{.python .input}
%%tab mxnet
np.zeros((2, 3, 4))
```

```{.python .input}
%%tab pytorch
torch.zeros((2, 3, 4))
```

```{.python .input}
%%tab tensorflow
tf.zeros((2, 3, 4))
```

Similarly, we can create a tensor 
with all ones by invoking `ones`.

```{.python .input}
%%tab mxnet
np.ones((2, 3, 4))
```

```{.python .input}
%%tab pytorch
torch.ones((2, 3, 4))
```

```{.python .input}
%%tab tensorflow
tf.ones((2, 3, 4))
```

We often wish to 
[**sample each element randomly (and independently)**] 
from a given probability distribution.
For example, the parameters of neural networks
are often initialized randomly.
The following snippet creates a tensor 
with elements drawn from 
a standard Gaussian (normal) distribution
with mean 0 and standard deviation 1.

```{.python .input}
%%tab mxnet
np.random.normal(0, 1, size=(3, 4))
```

```{.python .input}
%%tab pytorch
torch.randn(3, 4)
```

```{.python .input}
%%tab tensorflow
tf.random.normal(shape=[3, 4])
```

Finally, we can construct tensors by
[**supplying the exact values for each element**] 
by supplying (possibly nested) Python list(s) 
containing numerical literals.
Here, we construct a matrix with a list of lists,
where the outermost list corresponds to axis 0,
and the inner list to axis 1.

```{.python .input}
%%tab mxnet
np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
%%tab pytorch
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
%%tab tensorflow
tf.constant([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

## Indexing and Slicing

As with  Python lists,
we can access tensor elements 
by indexing (starting with 0).
To access an element based on its position
relative to the end of the list,
we can use negative indexing.
Finally, we can access whole ranges of indices 
via slicing (e.g., `X[start:stop]`), 
where the returned value includes 
the first index (`start`) *but not the last* (`stop`).
Finally, when only one index (or slice)
is specified for a $k^\mathrm{th}$ order tensor,
it is applied along axis 0.
Thus, in the following code,
[**`[-1]` selects the last row and `[1:3]`
selects the second and third rows**].

```{.python .input}
%%tab all
X[-1], X[1:3]
```

:begin_tab:`mxnet, pytorch`
Beyond reading, (**we can also write elements of a matrix by specifying indices.**)
:end_tab:

:begin_tab:`tensorflow`
`Tensors` in TensorFlow are immutable, and cannot be assigned to.
`Variables` in TensorFlow are mutable containers of state that support
assignments. Keep in mind that gradients in TensorFlow do not flow backwards
through `Variable` assignments.

Beyond assigning a value to the entire `Variable`, we can write elements of a
`Variable` by specifying indices.
:end_tab:

```{.python .input}
%%tab mxnet, pytorch
X[1, 2] = 17
X
```

```{.python .input}
%%tab tensorflow
X_var = tf.Variable(X)
X_var[1, 2].assign(9)
X_var
```

If we want [**to assign multiple elements the same value,
we apply the indexing on the left-hand side 
of the assignment operation.**]
For instance, `[:2, :]`  accesses 
the first and second rows,
where `:` takes all the elements along axis 1 (column).
While we discussed indexing for matrices,
this also works for vectors
and for tensors of more than 2 dimensions.

```{.python .input}
%%tab mxnet, pytorch
X[:2, :] = 12
X
```

```{.python .input}
%%tab tensorflow
X_var = tf.Variable(X)
X_var[:2, :].assign(tf.ones(X_var[:2,:].shape, dtype=tf.float32) * 12)
X_var
```

## Operations

Now that we know how to construct tensors
and how to read from and write to their elements,
we can begin to manipulate them
with various mathematical operations.
Among the most useful tools 
are the *elementwise* operations.
These apply a standard scalar operation
to each element of a tensor.
For functions that take two tensors as inputs,
elementwise operations apply some standard binary operator
on each pair of corresponding elements.
We can create an elementwise function 
from any function that maps 
from a scalar to a scalar.

In mathematical notation, we denote such
*unary* scalar operators (taking one input)
by the signature 
$f: \mathbb{R} \rightarrow \mathbb{R}$.
This just means that the function maps
from any real number onto some other real number.
Most standard operators can be applied elementwise
including unary operators like $e^x$.

```{.python .input}
%%tab mxnet
np.exp(x)
```

```{.python .input}
%%tab pytorch
torch.exp(x)
```

```{.python .input}
%%tab tensorflow
tf.exp(x)
```

Likewise, we denote *binary* scalar operators,
which map pairs of real numbers
to a (single) real number
via the signature 
$f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$.
Given any two vectors $\mathbf{u}$ 
and $\mathbf{v}$ *of the same shape*,
and a binary operator $f$, we can produce a vector
$\mathbf{c} = F(\mathbf{u},\mathbf{v})$
by setting $c_i \gets f(u_i, v_i)$ for all $i$,
where $c_i, u_i$, and $v_i$ are the $i^\mathrm{th}$ elements
of vectors $\mathbf{c}, \mathbf{u}$, and $\mathbf{v}$.
Here, we produced the vector-valued
$F: \mathbb{R}^d, \mathbb{R}^d \rightarrow \mathbb{R}^d$
by *lifting* the scalar function
to an elementwise vector operation.
The common standard arithmetic operators
for addition (`+`), subtraction (`-`), 
multiplication (`*`), division (`/`), 
and exponentiation (`**`)
have all been *lifted* to elementwise operations
for identically-shaped tensors of arbitrary shape.

```{.python .input}
%%tab mxnet
x = np.array([1, 2, 4, 8])
y = np.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

```{.python .input}
%%tab pytorch
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

```{.python .input}
%%tab tensorflow
x = tf.constant([1.0, 2, 4, 8])
y = tf.constant([2.0, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

In addition to elementwise computations,
we can also perform linear algebra operations,
such as dot products and matrix multiplications.
We will elaborate on these shortly
in :numref:`sec_linear-algebra`.

We can also [***concatenate* multiple tensors together,**]
stacking them end-to-end to form a larger tensor.
We just need to provide a list of tensors
and tell the system along which axis to concatenate.
The example below shows what happens when we concatenate
two matrices along rows (axis 0)
vs. columns (axis 1).
We can see that the first output's axis-0 length ($6$)
is the sum of the two input tensors' axis-0 lengths ($3 + 3$);
while the second output's axis-1 length ($8$)
is the sum of the two input tensors' axis-1 lengths ($4 + 4$).

```{.python .input}
%%tab mxnet
X = np.arange(12).reshape(3, 4)
Y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
np.concatenate([X, Y], axis=0), np.concatenate([X, Y], axis=1)
```

```{.python .input}
%%tab pytorch
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
```

```{.python .input}
%%tab tensorflow
X = tf.reshape(tf.range(12, dtype=tf.float32), (3, 4))
Y = tf.constant([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
tf.concat([X, Y], axis=0), tf.concat([X, Y], axis=1)
```

Sometimes, we want to 
[**construct a binary tensor via *logical statements*.**]
Take `X == Y` as an example.
For each position `i, j`, if `X[i, j]` and `Y[i, j]` are equal, 
then the corresponding entry in the result takes value `1`,
otherwise it takes value `0`.

```{.python .input}
%%tab all
X == Y
```

[**Summing all the elements in the tensor**] yields a tensor with only one element.

```{.python .input}
%%tab mxnet, pytorch
X.sum()
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(X)
```

## Broadcasting
:label:`subsec_broadcasting`

By now, you know how to perform 
elementwise binary operations
on two tensors of the same shape. 
Under certain conditions,
even when shapes differ, 
we can still [**perform elementwise binary operations
by invoking the *broadcasting mechanism*.**]
Broadcasting works according to 
the following two-step procedure:
(i) expand one or both arrays
by copying elements along axes with length 1
so that after this transformation,
the two tensors have the same shape;
(ii) perform an elementwise operation
on the resulting arrays.

```{.python .input}
%%tab mxnet
a = np.arange(3).reshape(3, 1)
b = np.arange(2).reshape(1, 2)
a, b
```

```{.python .input}
%%tab pytorch
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
```

```{.python .input}
%%tab tensorflow
a = tf.reshape(tf.range(3), (3, 1))
b = tf.reshape(tf.range(2), (1, 2))
a, b
```

Since `a` and `b` are $3\times1$ 
and $1\times2$ matrices, respectively,
their shapes do not match up.
Broadcasting produces a larger $3\times2$ matrix 
by replicating matrix `a` along the columns
and matrix `b` along the rows
before adding them elementwise.

```{.python .input}
%%tab all
a + b
```

## Saving Memory

[**Running operations can cause new memory to be
allocated to host results.**]
For example, if we write `Y = X + Y`,
we dereference the tensor that `Y` used to point to
and instead point `Y` at the newly allocated memory.
We can demonstrate this issue with Python's `id()` function,
which gives us the exact address 
of the referenced object in memory.
Note that after we run `Y = Y + X`,
`id(Y)` points to a different location.
That's because Python first evaluates `Y + X`,
allocating new memory for the result 
and then points `Y` to this new location in memory.

```{.python .input}
%%tab all
before = id(Y)
Y = Y + X
id(Y) == before
```

This might be undesirable for two reasons.
First, we do not want to run around
allocating memory unnecessarily all the time.
In machine learning, we often have
hundreds of megabytes of parameters
and update all of them multiple times per second.
Whenever possible, we want to perform these updates *in place*.
Second, we might point at the 
same parameters from multiple variables.
If we do not update in place, 
we must be careful to update all of these references,
lest we spring a memory leak 
or inadvertently refer to stale parameters.

:begin_tab:`mxnet, pytorch`
Fortunately, (**performing in-place operations**) is easy.
We can assign the result of an operation
to a previously allocated array `Y`
by using slice notation: `Y[:] = <expression>`.
To illustrate this concept, 
we overwrite the values of tensor `Z`,
after initializing it, using `zeros_like`,
to have the same shape as `Y`.
:end_tab:

:begin_tab:`tensorflow`
`Variables` are mutable containers of state in TensorFlow. They provide
a way to store your model parameters.
We can assign the result of an operation
to a `Variable` with `assign`.
To illustrate this concept, 
we overwrite the values of `Variable` `Z`
after initializing it, using `zeros_like`,
to have the same shape as `Y`.
:end_tab:

```{.python .input}
%%tab mxnet
Z = np.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
%%tab pytorch
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
%%tab tensorflow
Z = tf.Variable(tf.zeros_like(Y))
print('id(Z):', id(Z))
Z.assign(X + Y)
print('id(Z):', id(Z))
```

:begin_tab:`mxnet, pytorch`
[**If the value of `X` is not reused in subsequent computations,
we can also use `X[:] = X + Y` or `X += Y`
to reduce the memory overhead of the operation.**]
:end_tab:

:begin_tab:`tensorflow`
Even once you store state persistently in a `Variable`, 
you may want to reduce your memory usage further by avoiding excess
allocations for tensors that are not your model parameters.
Because TensorFlow `Tensors` are immutable 
and gradients do not flow through `Variable` assignments, 
TensorFlow does not provide an explicit way to run
an individual operation in-place.

However, TensorFlow provides the `tf.function` decorator 
to wrap computation inside of a TensorFlow graph 
that gets compiled and optimized before running.
This allows TensorFlow to prune unused values, 
and to reuse prior allocations that are no longer needed. 
This minimizes the memory overhead of TensorFlow computations.
:end_tab:

```{.python .input}
%%tab mxnet, pytorch
before = id(X)
X += Y
id(X) == before
```

```{.python .input}
%%tab tensorflow
@tf.function
def computation(X, Y):
    Z = tf.zeros_like(Y)  # This unused value will be pruned out
    A = X + Y  # Allocations will be reused when no longer needed
    B = A + Y
    C = B + Y
    return C + Y

computation(X, Y)
```

## Conversion to Other Python Objects

:begin_tab:`mxnet, tensorflow`
[**Converting to a NumPy tensor (`ndarray`)**], or vice versa, is easy.
The converted result does not share memory.
This minor inconvenience is actually quite important:
when you perform operations on the CPU or on GPUs,
you do not want to halt computation, waiting to see
whether the NumPy package of Python 
might want to be doing something else
with the same chunk of memory.
:end_tab:

:begin_tab:`pytorch`
[**Converting to a NumPy tensor (`ndarray`)**], or vice versa, is easy.
The torch Tensor and numpy array 
will share their underlying memory, 
and changing one through an in-place operation 
will also change the other.
:end_tab:

```{.python .input}
%%tab mxnet
A = X.asnumpy()
B = np.array(A)
type(A), type(B)
```

```{.python .input}
%%tab pytorch
A = X.numpy()
B = torch.from_numpy(A)
type(A), type(B)
```

```{.python .input}
%%tab tensorflow
A = X.numpy()
B = tf.constant(A)
type(A), type(B)
```

To (**convert a size-1 tensor to a Python scalar**),
we can invoke the `item` function or Python's built-in functions.

```{.python .input}
%%tab mxnet
a = np.array([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
%%tab pytorch
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
%%tab tensorflow
a = tf.constant([3.5]).numpy()
a, a.item(), float(a), int(a)
```

## Summary

 * The tensor class is the main interface for storing and manipulating data in deep learning libraries.
 * Tensors provide a variety of functionalities including construction routines; indexing and slicing; basic mathematics operations; broadcasting; memory-efficient assignment; and conversion to and from other Python objects.


## Exercises

1. Run the code in this section. Change the conditional statement `X == Y` to `X < Y` or `X > Y`, and then see what kind of tensor you can get.
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
