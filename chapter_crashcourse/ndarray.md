# Data Manipulation
:label:`chapter_ndarray`

It is impossible to get anything done if we cannot manipulate data. Generally, there are two important things we need to do with data: (i) acquire it and (ii) process it once it is inside the computer. There is no point in acquiring data if we do not even know how to store it, so let's get our hands dirty first by playing with synthetic data. We will start by introducing the N-dimensional array (ndarray), MXNet's primary tool for storing and transforming data. 
If you have worked with NumPy, a scientific computing package of Python, you are ready to fly. You will notice that the MXNet's ndarray is an extension to NumPy's ndarray with a few key advantages. First, the former supports asynchronous computation on CPU, GPU, and distributed cloud architectures, whereas the latter only supports CPU computation. Second, MXNet's ndarray supports for automatic differentiation. These properties make MXNet's ndarray indispensable for deep learning.

## Getting Started

Throughout this chapter, we are aiming to get you up and running with the basic functionality. Do not worry if you do not understand all of the basic math, like element-wise operations or normal distributions. In the next two chapters we will take another pass at the same material with practical examples. On the other hand, if you want to go deeper into the mathematical content, see :numref:`chapter_math`.

We begin by importing `np` module and `npx` module from MXNet. Here, `np` module includes all the functions that NumPy has, while `npx` module contains all the other extended functions which empower deep learning computation. When using ndarray, we almost always invoke the `set_np` function: this is for compatibility of ndarray processing by other components of MXNet.

```{.python .input  n=1}
from mxnet import np, npx
npx.set_np()
```

An ndarray represents (possibly multi-dimensional) arrays of numerical values. An ndarray with one axis corresponds (in math-speak) to *vectors*. An ndarray with two axes corresponds to *matrices*. For arrays with more than two axes, mathematicians do not have special names---they simply call them *tensors*.

The simplest object we can create is a vector. To start, we can use `arange` to create a row vector `x` with 12 consecutive integers. Here `x` is a one-dimensional ndarray that stores in main memory for CPU computing by default.

```{.python .input  n=2}
x = np.arange(12)
x
```

We can get the ndarray shape through the `shape` property.

```{.python .input  n=8}
x.shape
```

We can also get the total number of elements in the ndarray through the `size` property. This is the product of the elements of the shape. Since we are dealing with a vector here, both are identical.

```{.python .input  n=9}
x.size
```

We use the `reshape` function to change the shape of one (possibly multi-dimensional) ndarray, to another that contains the same number of elements.
For example, we can transform the shape of our line vector `x` to (3, 4), which contains the same values but interprets them as a matrix containing 3 rows and 4 columns. Note that although the shape has changed, the elements in `x` have not. Moreover, the `size` remains the same.

```{.python .input  n=3}
x = x.reshape((3, 4))
x
```

Reshaping by manually specifying each of the dimensions can get annoying. Once we know one of the dimensions, why should we have to perform the division ourselves to determine the other? For example, above, to get a matrix with 3 rows, we had to specify that it should have 4 columns (to account for the 12 elements). Fortunately, ndarray can automatically work out one dimension given the other. We can invoke this capability by placing `-1` for the dimension that we would like ndarray to automatically infer. In our case, instead of
`x.reshape((3, 4))`, we could have equivalently used `x.reshape((-1, 4))` or `x.reshape((3, -1))`.

```{.python .input}
np.empty((3, 4))
```

The `empty` method just grabs some memory and hands us back a matrix without setting the values of any of its entries. This is very efficient but it means that the entries might take any arbitrary values, including very big ones! Typically, we'll want our matrices initialized either with ones, zeros, some known constant or numbers randomly sampled from a known distribution.

Perhaps most often, we want an array of all zeros. To create an ndarray representing a tensor with all elements set to 0 and a shape of (2, 3, 4) we can invoke:

```{.python .input  n=4}
np.zeros((2, 3, 4))
```

We can create tensors with each element set to 1 works via

```{.python .input  n=5}
np.ones((2, 3, 4))
```

We can also specify the value of each element in the desired ndarray by supplying a Python list containing the numerical values.

```{.python .input  n=6}
y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
y
```

In some cases, we will want to randomly sample the values of each element in the ndarray according to some known probability distribution. This is especially common when we intend to use the array as a parameter in a neural network. The following snippet creates an ndarray with a shape of (3,4). Each of its elements is randomly sampled in a normal distribution with zero mean and one standard deviation.

```{.python .input  n=7}
np.random.normal(0, 1, size=(3, 4))
```

## Operations

Oftentimes, we want to apply functions to arrays. Some of the simplest and most useful functions are the element-wise functions. These operate by performing a single scalar operation on the corresponding elements of two arrays. We can create an element-wise function from any function that maps from the scalars to the scalars. In math notations we would denote such a function as $f: \mathbb{R} \rightarrow \mathbb{R}$. Given any two vectors $\mathbf{u}$ and $\mathbf{v}$ *of the same shape*, and the function f,
we can produce a vector $\mathbf{c} = F(\mathbf{u},\mathbf{v})$ by setting $c_i \gets f(u_i, v_i)$ for all $i$. Here, we produced the vector-valued $F: \mathbb{R}^d \rightarrow \mathbb{R}^d$ by *lifting* the scalar function to an element-wise vector operation. In MXNet, the common standard arithmetic operators (+,-,/,\*,\*\*) have all been *lifted* to element-wise operations for identically-shaped tensors of arbitrary shape. We can call element-wise operations on any two tensors of the same shape, including matrices.

```{.python .input}
x = np.array([1, 2, 4, 8])
y = np.ones_like(x) * 2
print('x =', x)
print('x + y', x + y)
print('x - y', x - y)
print('x * y', x * y)
print('x ** y', x ** y)
print('x / y', x / y)
```

Many more operations can be applied element-wise, such as exponentiation:

```{.python .input  n=12}
np.exp(x)
```

In addition to computations by element, we can also perform matrix operations, like matrix multiplication using the `dot` function. Next, we will perform matrix multiplication of `x` and the transpose of `y`. We define `x` as a matrix of 3 rows and 4 columns, and `y` is transposed into a matrix of 4 rows and 3 columns. The two matrices are multiplied to obtain a matrix of 3 rows and 3 columns (if you are confused about what this means, do not worry - we will explain matrix operations in much more detail in :numref:`chapter_linear_algebra`).

```{.python .input  n=13}
x = np.arange(12).reshape((3,4))
y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
np.dot(x, y.T)
```

We can also merge multiple ndarrays. For that, we need to tell the system along which dimension to merge. The example below merges two matrices along dimension 0 (along rows) and dimension 1 (along columns) respectively.

```{.python .input}
np.concatenate([x, y], axis=0), np.concatenate([x, y], axis=1)
```

Sometimes, we may want to construct binary ndarrays via logical statements. Take `x == y` as an example. If `x` and `y` are equal for some entry, the new ndarray has a value of 1 at the same position; otherwise it is 0.

```{.python .input}
x == y
```

Summing all the elements in the ndarray yields an ndarray with only one element.

```{.python .input}
x.sum()
```

For stylistic convenience, we can write `y.exp()`, `x.sum()`, `x.norm()`, etc. also as `np.exp(y)`, `np.sum(x)`, `np.linalg.norm(x)`.

## Broadcast Mechanism

In the above section, we saw how to perform operations on two ndarrays of the same shape. When their shapes differ, a broadcasting mechanism may be triggered: first, copy the elements appropriately so that the two ndarrays have the same shape, and then carry out operations by element.

```{.python .input  n=14}
a = np.arange(3).reshape((3, 1))
b = np.arange(2).reshape((1, 2))
print('a : ', a)
print('b : ', b)
```

Since `a` and `b` are (3x1) and (1x2) matrices respectively, their shapes do not match up if we want to add them. We 'broadcast' the entries of both matrices into a larger (3x2) matrix as follows: for matrix `a` it replicates the columns, for matrix `b` it replicates the rows before adding up both element-wise.

```{.python .input}
a + b
```

## Indexing and Slicing

Just like in any other Python array, elements in an ndarray can be accessed by its index. In good Python tradition the first element has index 0 and ranges are specified to include the first but not the last element. 

By this logic, `[-1]` selects the last element and `[1:3]` selects the second and the third elements. 
Notice that if you specify an integer index of an ndarray, it will return a scalar.
However, if you specify an array of indices as the slicing range, it will return an array of scalars. Let's try this out and compare the outputs.

```{.python .input  n=19}
print('x[-1] : ', x[-1])
print('x[1:3] : ', x[1:3])

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

In the previous example, every time we ran an operation, we allocated new memory to host its results. For example, if we write `y = x + y`, we will dereference the matrix that `y` used to point to and instead point it at the newly allocated memory. In the following example we demonstrate this with Python's `id()` function, which gives us the exact address of the referenced object in memory. After running `y = y + x`, we will find that `id(y)` points to a different location. That is because Python first evaluates `y + x`, allocating new memory for the result and then subsequently redirects `y` to point at this new location in memory.

```{.python .input  n=15}
before = id(y)
y = y + x
id(y) == before
```

This might be undesirable for two reasons. First, we do not want to run around allocating memory unnecessarily all the time. In machine learning, we might have hundreds of megabytes of parameters and update all of them multiple times per second. Typically, we will want to perform these updates *in place*. Second, we might point at the same parameters from multiple variables. If we do not update in place, this could cause a memory leak, making it possible for us to inadvertently reference stale parameters.

Fortunately, performing in-place operations in MXNet is easy. We can assign the result of an operation to a previously allocated array with slice notation, e.g., `y[:] = <expression>`. To illustrate the behavior, we first clone the shape of a matrix using `zeros_like` to allocate a block of 0 entries.

```{.python .input  n=16}
z = np.zeros_like(y)  
print('id(z):', id(z))
z[:] = x + y
print('id(z):', id(z))
```

If the value of `x ` is not reused in subsequent computations, we can also use `x[:] = x + y` or `x += y` to reduce the memory overhead of the operation.

```{.python .input  n=18}
before = id(x)
x += y
id(x) == before
```

## `mxnet.numpy.ndarray` and `numpy.ndarray`

Transforming an ndarray from an object in NumPy (a scientific computing package of Python) to an object in MXNet package, or *vice versa*, is easy. The converted array does not share memory. This minor inconvenience is actually quite important: when you perform operations on the CPU or one of the GPUs, you do not want MXNet having to wait whether NumPy might want to be doing something else with the same chunk of memory. The  `array` and `asnumpy` functions do the trick.

```{.python .input  n=22}
a = x.asnumpy()
print(type(a))
b = np.array(a)
print(type(b))
```

## Exercises

1. Run the code in this section. Change the conditional statement `x == y` in this section to `x < y` or `x > y`, and then see what kind of ndarray you can get.
1. Replace the two ndarrays that operate by element in the broadcast mechanism with other shapes, e.g. three dimensional tensors. Is the result the same as expected?
1. Assume that we have three matrices `a`, `b` and `c`. Rewrite `c = np.linalg.dot(a, b.T) + c` in the most memory efficient manner.

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2316)

![](../img/qr_ndarray.svg)
