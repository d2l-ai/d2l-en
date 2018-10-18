# Data Operations

In deep learning, we frequently operate on data. To experience hands-on deep learning, this section describes how to operate on data in the memory.

In MXNet, NDArray is the primary tool for storing and transforming data. If you have used NumPy before, you will find that NDArray is very similar to NumPy's multidimensional array. However, NDArray provides more features, such as GPU computing and auto-derivation, which makes it more suitable for deep learning.


## Create NDArray

Let us introduce the most basic functionalities of NDArray first. If you are not familiar with the mathematical operations we use, you can refer to the [“Mathematical Basics”](../chapter_appendix/math.md) section in the appendix.

First, import the `ndarray` module from MXNet. Here, `nd` is short for `ndarray`.

```{.python .input  n=1}
from mxnet import nd
```

Then, we create a row vector using the `arrange` function.

```{.python .input  n=2}
x = nd.arange(12)
x
```

This returns an NDArray instance containing 12 consecutive integers starting from 0. From the property `<NDArray 12 @cpu(0)>` shown when printing `x` we can see that it is a one-dimensional array with a length of 12 and is created in the CPU main memory. The 0 in "@cpu(0)" has no special meaning and does not represent a specific core.

We can get the NDArray instance shape through the `shape` property.

```{.python .input  n=8}
x.shape
```

We can also get the total number of elements in the NDArray instance through the `size` property.

```{.python .input  n=9}
x.size
```

In the following, we use the `reshape` function to change the shape of the line vector `x` to (3, 4), which is a matrix of 3 rows and 4 columns. Except for the shape change, the elements in`x` remain unchanged.

```{.python .input  n=3}
x = x.reshape((3, 4))
x
```

Notice that the shape in the `x` property has changed. The above `x.reshape((3, 4))` can also be written as `x.reshape((-1, 4))` or `x.reshape((3, -1))`. Since the number of elements of `x` is known, here `-1` can be inferred from the number of elements and the size of other dimensions.

Next, we create a tensor with each element being 0 and a shape of (2, 3, 4). In fact, the previously created vectors and matrices are special tensors.

```{.python .input  n=4}
nd.zeros((2, 3, 4))
```

Similarly, we can create a tensor with each element being 1.

```{.python .input  n=5}
nd.ones((3, 4))
```

We can also specify the value of each element in the NDArray that needs to be created through a Python list.

```{.python .input  n=6}
y = nd.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
y
```

In some cases, we need to randomly generate the value of each element in the NDArray. Next, we create an NDArray with a shape of (3,4). Each of its elements is randomly sampled in a normal distribution with a mean of 0 and standard deviation of 1.

```{.python .input  n=7}
nd.random.normal(0, 1, shape=(3, 4))
```

## Operation

NDArray supports a large number of operators. For example, we can carry out addition by element on two previously created NDArrays with a shape of (3, 4). The shape of the result does not change.

```{.python .input  n=10}
x + y
```

Multiply by element:

```{.python .input  n=11}
x * y
```

Divide by element:

```{.python .input}
x / y
```

Index operations by element:

```{.python .input  n=12}
y.exp()
```

In addition to computations by element, we can also use the `dot` function for matrix operations. Next, we will perform matrix multiplication to transpose `x` and `y`. Since `x` is a matrix of 3 rows and 4 columns, `y` is transposed into a matrix of 4 rows and 3 columns. The two matrices are multiplied to obtain a matrix of 3 rows and 3 columns.

```{.python .input  n=13}
nd.dot(x, y.T)
```

We can also merge multiple NDArrays. Next, we concatenate two matrices on the line (dimension 0, the leftmost element in the shape) and the column (dimension 1, the second element from the left in the shape).

```{.python .input}
nd.concat(x, y, dim=0), nd.concat(x, y, dim=1)
```

A new NDArray with an element of 0 or 1 can be obtained using the conditional judgment. Take `x == y` as an example. If `x` and `y` are determined to be true at the same position (value is equal), then the new NDArray has a value of 1 at the same position; otherwise, it is 0.

```{.python .input}
x == y
```

Summing all the elements in the NDArray yields an NDArray with only one element.

```{.python .input}
x.sum()
```

We can transform the result into a scalar in Python using the `asscalar` function. In the following example, the $L_2$ norm result of `x` is a single element NDArray, the same as the previous example, but the final result is transformed into a scalar in Python.

```{.python .input}
x.norm().asscalar()
```

We can also rewrite `y.exp()`, `x.sum()`, `x.norm()`, etc. as `nd.exp(y)`, `nd.sum(x)`, `nd.norm(x)`, etc.

## Broadcast Mechanism

In the above section, we saw how to perform operations by element on two NDArrays of the same shape. When two elements of different shapes of NDArray are operated by element, a broadcasting mechanism may be triggered. First, copy the elements appropriately so that the two NDArrays have the same shape, and then carry out operations by element.

Define two NDArrays:

```{.python .input  n=14}
a = nd.arange(3).reshape((3, 1))
b = nd.arange(2).reshape((1, 2))
a, b
```

Since `a`和`b` is a matrix of 3 rows and 1 column, and 1 row and 2 columns respectively, if it is needed to compute `a+b`, then the three elements in the first column of `a`are broadcast (copied) to the second column, and the two elements in the first line of `b` are broadcast (copied) to the second and third lines. In this way, we can add two matrixes of 3 rows and 2 columns by element.

```{.python .input}
a + b
```

## Index

In NDArray, the index represents the position of the element. The index of the NDArray is incremented from 0. For example, the line indexes of a matrix of 3 rows and 2 columns are 0, 1, and 2 respectively, and column indexes are 0 and 1 respectively.

In the following example, we specify the row index interception range of NDArray as `[1:3]`. Following the convention of closing the left and opening the right for the specified range, it intercepts two rows of the matrix `x` with indexes 1 and 2.

```{.python .input  n=19}
x[1:3]
```

We can specify the location of the individual elements in the NDArray that need to be accessed, such as the index of the rows and columns in the matrix, and reassign the element.

```{.python .input  n=20}
x[1, 2] = 9
x
```

Of course, we can also intercept some of the elements and reassign them. In the following example, we reassign each column element with a row index of 1.

```{.python .input  n=21}
x[1:2, :] = 12
x
```

## Memory Overhead of the Operation

In the previous example, we opened new memory for each operation to store the result of the operation. For example, even with operations like `y = x + y`, we will create new memory and then point `y` to the new memory. To demonstrate this, we can use the `id` function that comes with Python: if the IDs of the two instances are the same, then they correspond to the same memory address; otherwise, they are different.

```{.python .input  n=15}
before = id(y)
y = y + x
id(y) == before
```

If we want to specify the result to a specific memory, we can use the index described earlier to perform the replacement. In the example below, we first create an NDArray with the same shape as `y` and an element of 0 through `zeros_like`, denoted as `z`. Next, we write the result of `x + y` into the memory corresponding to `z` through `[:]`.

```{.python .input  n=16}
z = y.zeros_like()
before = id(z)
z[:] = x + y
id(z) == before
```

In fact, in the above example, we still created temporary memory for `x + y` to store the computation results, then copy it to the memory corresponding to `z`. If we want to avoid this temporary memory overhead, we can use the `out` parameter in the operator’s full name function.

```{.python .input  n=17}
nd.elemwise_add(x, y, out=z)
id(z) == before
```

If the value of `x ` is not reused in subsequent programs, we can also use `x[:] = x + y` or `x += y` to reduce the memory overhead of the operation.

```{.python .input  n=18}
before = id(x)
x += y
id(x) == before
```

## Mutual Transformation of NDArray and NumPy 

We can use the `array` and `asnumpy` functions to transform data between NDArray and NumPy formats. Next, the NumPy instance is transformed into an NDArray instance.

```{.python .input  n=22}
import numpy as np

p = np.ones((2, 3))
d = nd.array(p)
d
```

Then, the NDArray instance is transformed into a NumPy instance.

```{.python .input}
d.asnumpy()
```

## Summary

* NDArray is a primary tool for storing and transforming data in MXNet.
* We can easily create, operate, and specify indexes on NDArray, as well as transform them from/to NumPy.


## exercise

* Run the code in this section. Change the conditional judgment `x == y` in this section to `x < y` or `x > y`, and then see what kind of NDArray you can get.
* Replace the two NDArrays that operate by element in the broadcast mechanism with other shapes. Is the result the same as expected?


## Scan the QR Code to Access [Discussions](https://discuss.gluon.ai/t/topic/745)

![](../img/qr_ndarray.svg)
