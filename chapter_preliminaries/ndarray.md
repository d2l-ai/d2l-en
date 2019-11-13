# Data Manipulation
:label:`sec_ndarray`

In order to get anything done, we must have some way to manipulate data. 
Generally, there are two important things we need to do with data: 
(i) acquire them and (ii) process them once they are inside the computer. 
There is no point in acquiring data if we do not even know how to store it, 
so let us get our hands dirty first by playing with synthetic data. 
We will start by introducing the $n$-dimensional array (`ndarray`), 
MXNet's primary tool for storing and transforming data.
In MXNet, `ndarray` is a class and we also call its instance
an `ndarray` for brevity.

If you have worked with NumPy, perhaps the most widely-used 
scientific computing package in Python, then you are ready to fly. 
In short, we designed MXNet's `ndarray` to be 
an extension to NumPy's `ndarray` with a few key advantages.
First, MXNet's `ndarray` supports asynchronous computation 
on CPU, GPU, and distributed cloud architectures, 
whereas the latter only supports CPU computation. 
Second, MXNet's `ndarray` supports automatic differentiation. 
These properties make MXNet's `ndarray` indispensable for deep learning.
Throughout the book, the term `ndarray` refers to MXNet's `ndarray`
unless otherwise stated.



## Getting Started

Throughout this chapter, our aim is to get you up and running,
equipping you with the the basic math and numerical computing tools
that you will be mastering throughout the course of the book. 
Do not worry if you are not completely comfortable 
with all of the mathematical concepts or library functions. 
In the following sections we will revisit the same material 
in the context practical examples. 
On the other hand, if you already have some background 
and want to go deeper into the mathematical content, just skip this section.

To start, we import the `np` (`numpy`) and `npx` (`numpy_extension`) modules from MXNet. 
Here, the `np` module includes the same functions supported by NumPy, 
while the `npx` module contains a set of extensions 
developed to empower deep learning within a NumPy-like environment. 
When using `ndarray`, we almost always invoke the `set_np` function: 
this is for compatibility of `ndarray` processing by other components of MXNet.

```{.python .input  n=1}
from mxnet import np, npx
npx.set_np()
```

An `ndarray` represents an array of numerical values, which are possibly multi-dimensional. 
With one axis, an `ndarray` corresponds (in math) to a *vector*. 
With two axes, an `ndarray` corresponds to a *matrix*. 
Arrays with more than two axes do not have special mathematical names---we simply call them *tensors*.

To start, we can use `arange` to create a row vector `x` 
containing the first $12$ integers starting with $0$, though they are created as floats by default.
Each of the values in an `ndarray` is called an *element* of the `ndarray`.
For instance, there are $12$ elements in the `ndarray` `x`.
Unless otherwise specified, a new `ndarray` 
will be stored in main memory and designated for CPU-based computation.

```{.python .input  n=2}
x = np.arange(12)
x
```

We can access an `ndarray`'s *shape* (the length along each axis)
by inspecting its `shape` property.

```{.python .input  n=3}
x.shape
```

If we just want to know the total number of elements in an `ndarray`,
i.e., the product of all of the shape elements, 
we can inspect its `size` property. 
Because we are dealing with a vector here, 
the single element of its `shape` is identical to its `size`.

```{.python .input  n=4}
x.size
```

To change the shape of an `ndarray` 
without altering either the number of elements or their values,
we can invoke the `reshape` function.
For example, we can transform our `ndarray`, `x`, 
from a row vector with shape ($12$,) to a matrix of shape ($3$, $4$).
This new `ndarray` contains the exact same values, and 
treats such values as a matrix organized as $3$ rows and $4$ columns. 
To reiterate, although the shape has changed, the elements in `x` have not. 
Consequently, the `size` remains the same.

```{.python .input  n=5}
x = x.reshape(3, 4)
x
```

Reshaping by manually specifying each of the dimensions can sometimes get annoying. 
For instance, if our target shape is a matrix with shape (height, width),
after we know the width, the height is given implicitly.
Why should we have to perform the division ourselves? 
In the example above, to get a matrix with $3$ rows, 
we specified both that it should have $3$ rows and $4$ columns. 
Fortunately, `ndarray` can automatically work out one dimension given the rest. 
We invoke this capability by placing `-1` for the dimension
that we would like `ndarray` to automatically infer. 
In our case, instead of calling `x.reshape(3, 4)`, 
we could have equivalently called `x.reshape(-1, 4)` or `x.reshape(3, -1)`.

The `empty` method grabs a chunk of memory and hands us back a matrix 
without bothering to change the value of any of its entries. 
This is remarkably efficient but we must be careful because 
the entries might take arbitrary values, including very big ones!

```{.python .input  n=6}
np.empty((3, 4))
```

Typically, we will want our matrices initialized either with ones, zeros, 
some known constants, or numbers randomly sampled from a known distribution.
Perhaps most often, we want an array of all zeros. 
To create an `ndarray` representing a tensor with all elements set to $0$ 
and a shape of ($2$, $3$, $4$) we can invoke

```{.python .input  n=7}
np.zeros((2, 3, 4))
```

We can create tensors with each element set to 1 as follows:

```{.python .input  n=8}
np.ones((2, 3, 4))
```

In some cases, we will want to randomly sample the values 
of all the elements in an `ndarray` according 
to some known probability distribution. 
One common case is when we construct an array 
to serve as a parameter in a neural network. 
The following snippet creates an `ndarray` with shape ($3$, $4$). 
Each of its elements is randomly sampled 
from a standard Gaussian (normal) distribution 
with a mean of $0$ and a standard deviation of $1$.

```{.python .input  n=10}
np.random.normal(0, 1, size=(3, 4))
```

We can also specify the value of each element in the desired `ndarray` by supplying a Python list containing the numerical values.

```{.python .input  n=9}
np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

## Operations

This book is not about Web development---it is
not enough to just read and write values.
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

In math notation, we would denote such 
a *unary* scalar operator (taking one input) 
by the signature
$f: \mathbb{R} \rightarrow \mathbb{R}$
and a *binary* scalar operator (taking two inputs)
by the signature
$f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$. 
Given any two vectors $\mathbf{u}$ and $\mathbf{v}$ *of the same shape*, 
and a binary operator $f$, we can produce a vector 
$\mathbf{c} = F(\mathbf{u},\mathbf{v})$ 
by setting $c_i \gets f(u_i, v_i)$ for all $i$,
where $c_i, u_i$, and $v_i$ are the $i^\mathrm{th}$ elements of vectors $\mathbf{c}, \mathbf{u}$, and $\mathbf{v}$.
Here, we produced the vector-valued 
$F: \mathbb{R}^d, \mathbb{R}^d \rightarrow \mathbb{R}^d$ 
by *lifting* the scalar function to an elementwise vector operation.

In MXNet, the common standard arithmetic operators (`+`, `-`, `*`, `/`, and `**`) 
have all been *lifted* to elementwise operations 
for any identically-shaped tensors of arbitrary shape. 
We can call elementwise operations on any two tensors 
of the same shape.
In the following example, we use commas to formulate a $5$-element tuple,
where each element is the result of an elementwise operation.

```{.python .input  n=11}
x = np.array([1, 2, 4, 8])
y = np.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # The ** operator is exponentiation
```

Many more operations can be applied elementwise, including unary operators like exponentiation.

```{.python .input  n=12}
np.exp(x)
```

In addition to elementwise computations, 
we can also perform linear algebra operations, 
including vector dot products and matrix multiplication.
We will explain the crucial bits of linear algebra 
(with no assumed prior knowledge) in :numref:`sec_reduction-norm`.

We can also *concatenate* multiple `ndarray`s together,
stacking them end-to-end to form a larger `ndarray`. 
We just need to provide a list of `ndarray`s 
and tell the system along which axis to concatenate. 
The example below shows what happens when we concatenate 
two matrices along rows (axis $0$, the first element of the shape)
vs. columns (axis $1$, the second element of the shape).
We can see that, the first output `ndarray`'s axis-$0$ length ($6$)
is the sum of the two input `ndarray`s' axis-$0$ lengths ($3 + 3$);
while the second output `ndarray`'s axis-$1$ length ($8$)
is the sum of the two input `ndarray`s' axis-$1$ lengths ($4 + 4$).

```{.python .input  n=14}
x = np.arange(12).reshape(3, 4)
y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
np.concatenate([x, y], axis=0), np.concatenate([x, y], axis=1)
```

Sometimes, we want to construct a binary `ndarray` via *logical statements*. 
Take `x == y` as an example. 
For each position, if `x` and `y` are equal at that position,
the corresponding entry in the new `ndarray` takes a value of $1$,
meaning that the logical statement `x == y` is true at that position;
otherwise that position takes $0$.

```{.python .input  n=15}
x == y
```

Summing all the elements in the `ndarray` yields an `ndarray` with only one element.

```{.python .input  n=16}
x.sum()
```

For stylistic convenience, we can write `x.sum()`as `np.sum(x)`.

## Broadcasting Mechanism

In the above section, we saw how to perform 
elementwise operations on two `ndarray`s of the same shape.
Under certain conditions, even when shapes differ,
we can still perform elementwise operations
by invoking the *broadcasting mechanism*.
These mechanisms work in the following way:
First, expand one or both arrays
by copying elements appropriately 
so that after this transformation, 
the two `ndarray`s have the same shape.
Second, carry out the elementwise operations
on the resulting arrays.

In most cases, we broadcast along an axis where an array
initially only has length $1$, such as in the following example:

```{.python .input  n=17}
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

```{.python .input  n=18}
a + b
```

## Indexing and Slicing

Just as in any other Python array, elements in an `ndarray` can be accessed by index. 
As in any Python array, the first element has index $0$
and ranges are specified to include the first but *before* the last element. 

By this logic, `[-1]` selects the last element and `[1:3]` 
selects the second and the third elements. 
Let us try this out and compare the outputs.

```{.python .input  n=19}
x[-1], x[1:3]
```

Beyond reading, we can also write elements of a matrix by specifying indices.

```{.python .input  n=20}
x[1, 2] = 9
x
```

If we want to assign multiple elements the same value, 
we simply index all of them and then assign them the value. 
For instance, `[0:2, :]` accesses the first and second rows,
where `:` takes all the elements along axis $1$ (column).
While we discussed indexing for matrices, 
this obviously also works for vectors 
and for tensors of more than $2$ dimensions.

```{.python .input  n=21}
x[0:2, :] = 12
x
```

## Saving Memory

In the previous example, every time we ran an operation,
we allocated new memory to host its results. 
For example, if we write `y = x + y`, 
we will dereference the `ndarray` that `y` used to point to 
and instead point `y` at the newly allocated memory. 
In the following example, we demonstrate this with Python's `id()` function, 
which gives us the exact address of the referenced object in memory. 
After running `y = y + x`, we will find 
that `id(y)` points to a different location. 
That is because Python first evaluates `y + x`, 
allocating new memory for the result 
and then redirects `y` 
to point at this new location in memory.

```{.python .input  n=22}
before = id(y)
y = y + x
id(y) == before
```

This might be undesirable for two reasons.
First, we do not want to run around 
allocating memory unnecessarily all the time. 
In machine learning, we might have 
hundreds of megabytes of parameters 
and update all of them multiple times per second. 
Typically, we will want to perform these updates *in place*. 
Second, we might point at the same parameters from multiple variables. 
If we do not update in place, this could cause that
discarded memory is not released,
and make it possible for parts of our code
to inadvertently reference stale parameters.

Fortunately, performing in-place operations in MXNet is easy. 
We can assign the result of an operation 
to a previously allocated array with slice notation, 
e.g., `y[:] = <expression>`. 
To illustrate this concept, we first create a new matrix `z`
with the same shape as another `y`, 
using `zeros_like` to allocate a block of $0$ entries.

```{.python .input  n=23}
z = np.zeros_like(y)
print('id(z):', id(z))
z[:] = x + y
print('id(z):', id(z))
```

If the value of `x` is not reused in subsequent computations, 
we can also use `x[:] = x + y` or `x += y`
to reduce the memory overhead of the operation.

```{.python .input  n=24}
before = id(x)
x += y
id(x) == before
```

## Conversion to Other Python Objects

Converting an MXNet's `ndarray` to an object in the NumPy package of Python, or vice versa, is easy.
The converted result does not share memory.
This minor inconvenience is actually quite important: 
when you perform operations on the CPU or on GPUs, 
you do not want MXNet to halt computation, waiting to see
whether the NumPy package of Python might want to be doing something else 
with the same chunk of memory. 
The `array` and `asnumpy` functions do the trick.

```{.python .input  n=25}
a = x.asnumpy()
b = np.array(a)
type(a), type(b)
```

To convert a size-$1$ `ndarray` to a Python scalar, we can invoke the `item` function or Python's built-in functions.

```{.python .input}
a = np.array([3.5])
a, a.item(), float(a), int(a)
```

## Summary

* MXNet's `ndarray` is an extension to NumPy's `ndarray` with a few key advantages that make the former indispensable for deep learning.
* MXNet's `ndarray` provides a variety of functionalities such as basic mathematics operations, broadcasting, indexing, slicing, memory saving, and conversion to other Python objects.


## Exercises

1. Run the code in this section. Change the conditional statement `x == y` in this section to `x < y` or `x > y`, and then see what kind of `ndarray` you can get.
1. Replace the two `ndarray`s that operate by element in the broadcasting mechanism with other shapes, e.g., three dimensional tensors. Is the result the same as expected?


## [Discussions](https://discuss.mxnet.io/t/2316)

![](../img/qr_ndarray.svg)
