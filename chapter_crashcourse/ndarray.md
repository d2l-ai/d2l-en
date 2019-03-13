# Data Manipulation

It's impossible to get anything done if we can't manipulate data. Generally, there are two important things we need to do with data: (i) acquire it and (ii) process it once it's inside the computer. There's no point in acquiring data if we don't even know how to store it, so let's get our hands dirty first by playing with synthetic data. We'll start by introducing the NDArray,
MXNet's primary tool for storing and transforming data. If you've worked with NumPy before, you'll notice that NDArrays are, by design, similar to NumPy's multi-dimensional array. However, they confer a few key advantages. First, NDArrays support asynchronous computation on CPU, GPU, and distributed cloud architectures. Second, they provide support for automatic differentiation. These properties make NDArray indispensable for deep learning.

## Getting Started

Throughout this chapter, we're aiming to get you up and running with the basic functionality. Don't worry if you don't understand all of the basic math,
like element-wise operations or normal distributions. In the next two chapters we'll take another pass at the same material, teaching the material in the context of practica examples. On the other hand, if you want to go deeper into the mathematical content, see the ["Math"](../chapter_appendix/math.md) section in the appendix.

We begin by importing MXNet and the `ndarray` module from MXNet.
Here, `nd` is short for `ndarray`.

```{.python .input  n=1}
import mxnet as mx
from mxnet import nd
```


NDArrays represent (possibly multi-dimensional) arrays of numerical values. NDArrays with one axis correspond (in math-speak) to *vectors*. NDArrays with two axes correspond to *matrices*. For arrays with more than two axes, mathematicians don't have special names---they simply call them *tensors*.

The simplest object we can create is a vector. To start, we can use `arange` to create a row vector with 12 consecutive integers.

```{.python .input  n=2}
x = nd.arange(12)
x
```

When we print `x`, we can observe the property `<NDArray 12 @cpu(0)>` listed, which indicates that `x` is a one-dimensional array of length 12 and that it resides in CPU main memory. The 0 in `@cpu(0)` has no special meaning and does not represent a specific core.

We can get the NDArray instance shape through the `shape` property.

```{.python .input  n=8}
x.shape
```

We can also get the total number of elements in the NDArray instance through the `size` property. This is the product of the elements of the shape. Since we are dealing with a vector here, both are identical.

```{.python .input  n=9}
x.size
```

We use the `reshape` function to change the shape of one (possibly multi-dimensional) array, to another that contains the same number of elements.
For example, we can transform the shape of our line vector `x` to (3, 4), which contains the same values but interprets them as a matrix containing 3 rows and 4 columns. Note that although the shape has changed, the elements in `x` have not. Moreover, the `size` remains the same.

```{.python .input  n=3}
x = x.reshape((3, 4))
x
```

Reshaping by manually specifying each of the dimensions can get annoying. Once we know one of the dimensions, why should we have to perform the division our selves to determine the other? For example, above, to get a matrix with 3 rows, we had to specify that it should have 4 columns (to account for the 12 elements). Fortunately, NDArray can automatically work out one dimension given the other. We can invoke this capability by placing `-1` for the dimension that we would like NDArray to automatically infer. In our case, instead of
`x.reshape((3, 4))`, we could have equivalently used `x.reshape((-1, 4))` or `x.reshape((3, -1))`.

```{.python .input}
nd.empty((3, 4))
```

The `empty` method just grabs some memory and hands us back a matrix without setting the values of any of its entries. This is very efficient but it means that the entries might take any arbitrary values, including very big ones! Typically, we'll want our matrices initialized either with ones, zeros, some known constant or numbers randomly sampled from a known distribution.

Perhaps most often, we want an array of all zeros. To create an NDArray representing a tensor with all elements set to 0 and a shape of (2, 3, 4) we can invoke:

```{.python .input  n=4}
nd.zeros((2, 3, 4))
```

We can create tensors with each element set to 1 works via

```{.python .input  n=5}
nd.ones((2, 3, 4))
```

We can also specify the value of each element in the desired NDArray by supplying a Python list containing the numerical values.

```{.python .input  n=6}
y = nd.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
y
```

In some cases, we will want to randomly sample the values of each element in the NDArray according to some known probability distribution. This is especially common when we intend to use the array as a parameter in a neural network. The following snippet creates an NDArray with a shape of (3,4). Each of its elements is randomly sampled in a normal distribution with zero mean and unit variance.

```{.python .input  n=7}
nd.random.normal(0, 1, shape=(3, 4))
```

## Operations

Oftentimes, we want to apply functions to arrays. Some of the simplest and most useful functions are the element-wise functions. These operate by performing a single scalar operation on the corresponding elements of two arrays. We can create an element-wise function from any function that maps from the scalars to the scalars. In math notations we would denote such a function as $f: \mathbb{R} \rightarrow \mathbb{R}$. Given any two vectors $\mathbf{u}$ and $\mathbf{v}$ *of the same shape*, and the function f,
we can produce a vector $\mathbf{c} = F(\mathbf{u},\mathbf{v})$ by setting $c_i \gets f(u_i, v_i)$ for all $i$. Here, we produced the vector-valued $F: \mathbb{R}^d \rightarrow \mathbb{R}^d$ by *lifting* the scalar function to an element-wise vector operation. In MXNet, the common standard arithmetic operators (+,-,/,\*,\*\*) have all been *lifted* to element-wise operations for identically-shaped tensors of arbitrary shape. We can call element-wise operations on any two tensors of the same shape, including matrices.

```{.python .input}
x = nd.array([1, 2, 4, 8])
y = nd.ones_like(x) * 2
print('x =', x)
print('x + y', x + y)
print('x - y', x - y)
print('x * y', x * y)
print('x / y', x / y)
```

Many more operations can be applied element-wise, such as exponentiation:

```{.python .input  n=12}
x.exp()
```

In addition to computations by element, we can also perform matrix operations, like matrix multiplication using the `dot` function. Next, we will perform matrix multiplication of `x` and the transpose of `y`. We define `x` as a matrix of 3 rows and 4 columns, and `y` is transposed into a matrix of 4 rows and 3 columns. The two matrices are multiplied to obtain a matrix of 3 rows and 3 columns (if you're confused about what this means, don't worry - we will explain matrix operations in much more detail in the chapter on [linear algebra](linear-algebra.md)).

```{.python .input  n=13}
x = nd.arange(12).reshape((3,4))
y = nd.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
nd.dot(x, y.T)
```

We can also merge multiple NDArrays. For that, we need to tell the system along which dimension to merge. The example below merges two matrices along dimension 0 (along rows) and dimension 1 (along columns) respectively.

```{.python .input}
nd.concat(x, y, dim=0)
nd.concat(x, y, dim=1)
```

Sometimes, we may want to construct binary NDArrays via logical statements. Take `x == y` as an example. If `x` and `y` are equal for some entry, the new NDArray has a value of 1 at the same position; otherwise it is 0.

```{.python .input}
x == y
```

Summing all the elements in the NDArray yields an NDArray with only one element.

```{.python .input}
x.sum()
```

We can transform the result into a scalar in Python using the `asscalar` function. In the following example, the $\ell_2$ norm of `x` yields a single element NDArray. The final result is transformed into a scalar.

```{.python .input}
x.norm().asscalar()
```

For stylistic convenience, we can write `y.exp()`, `x.sum()`, `x.norm()`, etc. also as `nd.exp(y)`, `nd.sum(x)`, `nd.norm(x)`.

## Broadcast Mechanism

In the above section, we saw how to perform operations on two NDArrays of the same shape. When their shapes differ, a broadcasting mechanism may be triggered analogous to NumPy: first, copy the elements appropriately so that the two NDArrays have the same shape, and then carry out operations by element.

```{.python .input  n=14}
a = nd.arange(3).reshape((3, 1))
b = nd.arange(2).reshape((1, 2))
a, b
```

Since `a` and `b` are (3x1) and (1x2) matrices respectively, their shapes do not match up if we want to add them. NDArray addresses this by 'broadcasting' the entries of both matrices into a larger (3x2) matrix as follows: for matrix `a` it replicates the columns, for matrix `b` it replicates the rows before adding up both element-wise.

```{.python .input}
a + b
```

## Indexing and Slicing

Just like in any other Python array, elements in an NDArray can be accessed by its index. In good Python tradition the first element has index 0 and ranges are specified to include the first but not the last element. By this logic `1:3` selects the second and third element. Let's try this out by selecting the respective rows in a matrix.

```{.python .input  n=19}
x[1:3]
```

Beyond reading, we can also write elements of a matrix.

```{.python .input  n=20}
x[1, 2] = 9
x
```

If we want to assign multiple elements the same value, we simply index all of them and then assign them the value. For instance, `[0:2, :]` accesses the first and second rows. While we discussed indexing for matrices, this obviously also works for vectors and for tensors of more than 2 dimensions.

```{.python .input  n=21}
x[0:2, :] = 12
x
```

## Saving Memory

In the previous example, every time we ran an operation, we allocated new memory to host its results. For example, if we write `y = x + y`, we will dereference the matrix that `y` used to point to and instead point it at the newly allocated memory. In the following example we demonstrate this with Python's `id()` function, which gives us the exact address of the referenced object in memory. After running `y = y + x`, we'll find that `id(y)` points to a different location. That's because Python first evaluates `y + x`, allocating new memory for the result and then subsequently redirects `y` to point at this new location in memory.

```{.python .input  n=15}
before = id(y)
y = y + x
id(y) == before
```

This might be undesirable for two reasons. First, we don't want to run around allocating memory unnecessarily all the time. In machine learning, we might have hundreds of megabytes of parameters and update all of them multiple times per second. Typically, we'll want to perform these updates *in place*. Second, we might point at the same parameters from multiple variables. If we don't update in place, this could cause a memory leak, making it possible for us to inadvertently reference stale parameters.

Fortunately, performing in-place operations in MXNet is easy. We can assign the result of an operation to a previously allocated array with slice notation, e.g., `y[:] = <expression>`. To illustrate the behavior, we first clone the shape of a matrix using `zeros_like` to allocate a block of 0 entries.

```{.python .input  n=16}
z = y.zeros_like()
print('id(z):', id(z))
z[:] = x + y
print('id(z):', id(z))
```

While this looks pretty, `x+y` here will still allocate a temporary buffer to store the result of `x+y` before copying it to `y[:]`. To make even better use of memory, we can directly invoke the underlying `ndarray` operation, in this case `elemwise_add`, avoiding temporary buffers. We do this by specifying the `out` keyword argument, which every `ndarray` operator supports:

```{.python .input  n=17}
before = id(z)
nd.elemwise_add(x, y, out=z)
id(z) == before
```

If the value of `x ` is not reused in subsequent computations, we can also use `x[:] = x + y` or `x += y` to reduce the memory overhead of the operation.

```{.python .input  n=18}
before = id(x)
x += y
id(x) == before
```

## Mutual Transformation of NDArray and NumPy

Converting MXNet NDArrays to and from NumPy is easy. The converted arrays do *not* share memory. This minor inconvenience is actually quite important: when you perform operations on the CPU or one of the GPUs, you don't want MXNet having to wait whether NumPy might want to be doing something else with the same chunk of memory. The  `array` and `asnumpy` functions do the trick.

```{.python .input  n=22}
import numpy as np

a = x.asnumpy()
print(type(a))
b = nd.array(a)
print(type(b))
```

## Exercises

1. Run the code in this section. Change the conditional statement `x == y` in this section to `x < y` or `x > y`, and then see what kind of NDArray you can get.
1. Replace the two NDArrays that operate by element in the broadcast mechanism with other shapes, e.g. three dimensional tensors. Is the result the same as expected?
1. Assume that we have three matrices `a`, `b` and `c`. Rewrite `c = nd.dot(a, b.T) + c` in the most memory efficient manner.

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2316)

![](../img/qr_ndarray.svg)
