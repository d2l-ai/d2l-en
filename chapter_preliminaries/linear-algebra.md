# Linear Algebra
:label:`sec_linear-algebra`

Now that we know how to load and manipulate data, 
let us briefly review the basic tools from linear algebra
that you will need to understand and implement
the models covered in this book. 

## Scalars

Depending on your past experience with linear algebra and machine learning, 
you might only be familiar with handling one number at a time. 
Even paying at a restaurant requires basic operations such as 
adding and multiplying pairs of numbers.

Formally, we call values consisting
of just one numerical quantity *scalars*.
For example, the temperature in Palo Alto is a balmy $72$ degrees Fahrenheit.
If you wanted to convert this value to Celsius
you would evaluate the expression $c = \frac{5}{9}(f - 32)$, setting $f$ to $72$.
In this equation, each of the terms---$5$, $9$, and $32$---are scalar *values*.
The placeholders $c$ and $f$ are called *variables*
and they represent unknown scalar values.

In this book, we adopt the notation
where scalar variables are denoted
by ordinary lower-cased letters (e.g., $x$, $y$, and $z$).
We denote the space of all (continuous) *real-valued* scalars by $\mathbb{R}$.
For expedience, we will punt on rigorous definitions
of what precisely *space* is,
but just remember for now that the expression $x \in \mathbb{R}$
is a formal way to say that $x$ is a real-valued scalar.
The symbol $\in$ (pronounced "in")
simply denotes membership in a set.
Analogously, we could write $x, y \in \{0, 1\}$
to state that $x$ and $y$ are numbers
whose value can only be $0$ or $1$.

(**A scalar is represented by a tensor with just one element.**)
In the next snippet, we instantiate two scalars
and perform some familiar arithmetic operations,
namely addition, multiplication, division, and exponentiation.

```{.python .input}
from mxnet import np, npx
npx.set_np()

x = np.array(3.0)
y = np.array(2.0)

x + y, x * y, x / y, x ** y
```

```{.python .input}
#@tab pytorch
import torch

x = torch.tensor(3.0)
y = torch.tensor(2.0)

x + y, x * y, x / y, x**y
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

x = tf.constant(3.0)
y = tf.constant(2.0)

x + y, x * y, x / y, x**y
```

## Vectors

[**You can think of a vector as simply a fixed-length list of scalar values.**]
We call these values the *elements* (*entries* or *components*) of the vector.
When our vectors represent examples from our dataset,
their values hold some real-world significance.
For example, if we were training a model to predict
the risk that a loan defaults,
we might associate each applicant with a vector
whose components correspond to their income,
length of employment, number of previous defaults, and other factors.
If we were studying the risk of heart attacks hospital patients potentially face,
we might represent each patient by a vector
whose components capture their most recent vital signs,
cholesterol levels, minutes of exercise per day, etc.
We will usually denote vectors as bold-faced,
lower-cased letters, such as $\mathbf{x}$, $\mathbf{y}$, and $\mathbf{z}$.

We work with vectors via one-dimensional tensors.
In general, tensors can have arbitrary but fixed lengths,
subject to the memory limits of your machine.

```{.python .input}
x = np.arange(3)
x
```

```{.python .input}
#@tab pytorch
x = torch.arange(3)
x
```

```{.python .input}
#@tab tensorflow
x = tf.range(3)
x
```

We can refer to an element of a vector by using a subscript.
For example, $x_2$ denotes the second element of $\mathbf{x}$. Since $x_2$ is a scalar,
we will not use bold-face when referring to it. A common convention for vectors is to write them as elements stacked vertically, also referred to as column notation.

$$\mathbf{x} =\begin{bmatrix}x_{1}  \\ \vdots  \\x_{n}\end{bmatrix},$$
:eqlabel:`eq_vec_def`

Here $x_1, \ldots, x_n$ are elements of the vector. Later on, we will also horizontally arranged vector entries, commonly known as row notation. In code,
we (**access any element by indexing into the tensor.**)

```{.python .input}
x[2]
```

```{.python .input}
#@tab pytorch
x[2]
```

```{.python .input}
#@tab tensorflow
x[2]
```

Let us revisit some concepts from :numref:`sec_ndarray`.
A vector is just an array of numbers.
And just as every array has a length, so does every vector.
If we want to say that a vector $\mathbf{x}$
consists of $n$ real-valued scalars,
we can express this as $\mathbf{x} \in \mathbb{R}^n$.
The length of a vector is commonly called the *dimension* of the vector.
As with an ordinary Python array,
we [**can access the length of a tensor**]
by calling Python's built-in `len()` function.

```{.python .input}
len(x)
```

```{.python .input}
#@tab pytorch
len(x)
```

```{.python .input}
#@tab tensorflow
len(x)
```

When a tensor represents a vector (with precisely one axis),
we can also access its length via the `.shape` attribute.
The shape is a tuple that lists the length (dimensionality)
along each axis of the tensor.
(**For tensors with just one axis, the shape has just one element.**)

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

Note that the word "dimension" tends to get overloaded
in these contexts and this tends to confuse people.
To clarify, we use the dimensionality of a *vector* or of an *axis*
to refer to its length, i.e., the number of elements of a vector or an axis.
However, we use the dimensionality of a tensor
to refer to the number of axes that a tensor has.
In this sense, the dimensionality of some axis of a tensor
will be the length of that axis.


## Matrices

Just as vectors generalize scalars from order zero to order one,
matrices generalize vectors from order one to order two.
Matrices, which we will typically denote with bold-faced, capital letters
(e.g., $\mathbf{X}$, $\mathbf{Y}$, and $\mathbf{Z}$),
are represented in code as tensors with two axes.
We use $\mathbf{A} \in \mathbb{R}^{m \times n}$
to express that the matrix $\mathbf{A}$ consists of $m$ rows and $n$ columns of real-valued scalars.
Visually, we can illustrate any matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ as a table,
where each element $a_{ij}$ belongs to the $i^{\mathrm{th}}$ row and $j^{\mathrm{th}}$ column:

$$\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \\ \end{bmatrix}.$$
:eqlabel:`eq_matrix_def`


For any $\mathbf{A} \in \mathbb{R}^{m \times n}$, the shape of $\mathbf{A}$
is ($m$, $n$) or $m \times n$.
Specifically, when a matrix has the same number of rows and columns,
its shape becomes a square; thus, it is called a *square matrix*.
We can [**create an $m \times n$ matrix**]
by specifying a shape with two components $m$ and $n$
when calling any of our favorite functions for instantiating a tensor.

```{.python .input}
A = np.arange(6).reshape(3, 2)
A
```

```{.python .input}
#@tab pytorch
A = torch.arange(6).reshape(3, 2)
A
```

```{.python .input}
#@tab tensorflow
A = tf.reshape(tf.range(6), (3, 2))
A
```

We can access the scalar element $a_{ij}$ of a matrix $\mathbf{A}$ in :eqref:`eq_matrix_def`
by specifying the indices for the row ($i$) and column ($j$),
such as $[\mathbf{A}]_{ij}$.
When the scalar elements of a matrix $\mathbf{A}$, such as in :eqref:`eq_matrix_def`, are not given,
we may simply use the lower-case letter of the matrix $\mathbf{A}$ with the index subscript, $a_{ij}$,
to refer to $[\mathbf{A}]_{ij}$.
To keep notation simple, commas are inserted to separate indices only when necessary,
such as $a_{2, 3j}$ and $[\mathbf{A}]_{2i-1, 3}$.


Sometimes, we want to flip the axes.
When we exchange a matrix's rows and columns,
the result is called the *transpose* of the matrix.
Formally, we signify a matrix $\mathbf{A}$'s transpose by $\mathbf{A}^\top$
and if $\mathbf{B} = \mathbf{A}^\top$, then $b_{ij} = a_{ji}$ for any $i$ and $j$.
Thus, the transpose of $\mathbf{A}$ in :eqref:`eq_matrix_def` is
a $n \times m$ matrix:

$$
\mathbf{A}^\top =
\begin{bmatrix}
    a_{11} & a_{21} & \dots  & a_{m1} \\
    a_{12} & a_{22} & \dots  & a_{m2} \\
    \vdots & \vdots & \ddots  & \vdots \\
    a_{1n} & a_{2n} & \dots  & a_{mn}
\end{bmatrix}.
$$

Now we access a (**matrix's transpose**) in code.

```{.python .input}
A.T
```

```{.python .input}
#@tab pytorch
A.T
```

```{.python .input}
#@tab tensorflow
tf.transpose(A)
```

As a special type of the square matrix,
[**a *symmetric matrix* $\mathbf{A}$ is equal to its transpose:
$\mathbf{A} = \mathbf{A}^\top$.**]

```{.python .input}
A = np.array([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A
```

```{.python .input}
#@tab pytorch
A = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A
```

```{.python .input}
#@tab tensorflow
A = tf.constant([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A
```

Now we compare `A` to its transpose.

```{.python .input}
A == A.T
```

```{.python .input}
#@tab pytorch
A == A.T
```

```{.python .input}
#@tab tensorflow
A == tf.transpose(A)
```

Matrices are useful data structures to organize data. 
For example, rows in our matrix might correspond to different houses (data examples),
while columns might correspond to different attributes.
This should sound familiar if you used spreadsheets before or
read :numref:`sec_pandas`. In this view each column corresponds to an attribute vector 
and each row vector is a data example.
Organizing data by row vectors is common in deep learning practice. 
For example, along the outermost axis of a tensor,
we can access or enumerate minibatches of data examples,
or just data examples if no minibatch exists.


## Tensors

Just as vectors generalize scalars, and matrices generalize vectors, we can build data structures with even more axes.
[**Tensors**]
("tensors" in this subsection refer to algebraic objects)
(**give us a generic way of describing $n$-dimensional arrays with an arbitrary number of axes.**)
Vectors, for example, are first-order tensors, and matrices are second-order tensors.
Tensors are denoted with capital letters of a special font face
(e.g., $\mathsf{X}$, $\mathsf{Y}$, and $\mathsf{Z}$)
and their indexing mechanism (e.g., $x_{ijk}$ and $[\mathsf{X}]_{1, 2i-1, 3}$) is similar to that of matrices.

Tensors will become more important when we start working with images,
 which arrive as $n$-dimensional arrays with 3 axes corresponding to the height, width, and a *channel* axis for stacking the color channels (red, green, and blue). For now, we will skip over higher order tensors and focus on the basics.

```{.python .input}
np.arange(24).reshape(2, 3, 4)
```

```{.python .input}
#@tab pytorch
torch.arange(24).reshape(2, 3, 4)
```

```{.python .input}
#@tab tensorflow
tf.reshape(tf.range(24), (2, 3, 4))
```

## Basic Properties of Tensor Arithmetic

Scalars, vectors, matrices, and tensors ("tensors" in this section refer to algebraic objects)
of an arbitrary number of axes
have some handy properties. For example, you might have noticed
that any elementwise unary operation does not change the shape of its operand.
Similarly,
[**given any two tensors with the same shape,
the result of any binary elementwise operation
will be a tensor of that same shape.**]
For example, adding two matrices of the same shape
performs elementwise addition over these two matrices.

```{.python .input}
A = np.arange(6).reshape(2, 3)
B = A.copy()  # Assign a copy of `A` to `B` by allocating new memory
A, A + B
```

```{.python .input}
#@tab pytorch
A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
B = A.clone()  # Assign a copy of `A` to `B` by allocating new memory
A, A + B
```

```{.python .input}
#@tab tensorflow
A = tf.reshape(tf.range(6, dtype=tf.float32), (2, 3))
B = A  # No cloning of `A` to `B` by allocating new memory
A, A + B
```

Specifically,
[**elementwise multiplication of two matrices is called their *Hadamard product***]
(math notation $\odot$).
Consider matrix $\mathbf{B} \in \mathbb{R}^{m \times n}$ whose element of row $i$ and column $j$ is $b_{ij}$. The Hadamard product of matrices $\mathbf{A}$ (defined in :eqref:`eq_matrix_def`) and $\mathbf{B}$

$$
\mathbf{A} \odot \mathbf{B} =
\begin{bmatrix}
    a_{11}  b_{11} & a_{12}  b_{12} & \dots  & a_{1n}  b_{1n} \\
    a_{21}  b_{21} & a_{22}  b_{22} & \dots  & a_{2n}  b_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1}  b_{m1} & a_{m2}  b_{m2} & \dots  & a_{mn}  b_{mn}
\end{bmatrix}.
$$

```{.python .input}
A * B
```

```{.python .input}
#@tab pytorch
A * B
```

```{.python .input}
#@tab tensorflow
A * B
```

[**Multiplying or adding a tensor to a scalar**] also leaves the shape of the tensor unchanged,
where each element of the operand tensor will be added or multiplied by the scalar.

```{.python .input}
a = 2
X = np.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
#@tab pytorch
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
#@tab tensorflow
a = 2
X = tf.reshape(tf.range(24), (2, 3, 4))
a + X, (a * X).shape
```

## Reduction
:label:`subseq_lin-alg-reduction`

One useful operation that we can perform with arbitrary tensors
is to
calculate [**the sum of their elements.**]
To express the sum of the elements in a vector $\mathbf{x}$ of length $n$,
we write $\sum_{i=1}^n x_i$. There's a simple function for it:

```{.python .input}
x = np.arange(3)
x, x.sum()
```

```{.python .input}
#@tab pytorch
x = torch.arange(3, dtype=torch.float32)
x, x.sum()
```

```{.python .input}
#@tab tensorflow
x = tf.range(3, dtype=tf.float32)
x, tf.reduce_sum(x)
```

We can express [**sums over the elements of tensors of arbitrary shape.**]
For example, the sum of the elements of an $m \times n$ matrix $\mathbf{A}$ could be written $\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$.

```{.python .input}
A.shape, A.sum()
```

```{.python .input}
#@tab pytorch
A.shape, A.sum()
```

```{.python .input}
#@tab tensorflow
A.shape, tf.reduce_sum(A)
```

By default, invoking the function for calculating the sum
*reduces* a tensor along all its axes to a scalar.
We can also [**specify the axes along which the tensor is reduced via summation.**]
Take matrices as an example.
To sum over all elements along the row dimension (axis 0) we specify `axis=0` in `sum`.
Since the input matrix reduces along axis 0 to generate the output vector,
the dimension of axis 0 of the input is omitted in the output.

```{.python .input}
A_sum_axis0 = A.sum(axis=0)
A_sum_axis0, A_sum_axis0.shape
```

```{.python .input}
#@tab pytorch
A_sum_axis0 = A.sum(axis=0)
A_sum_axis0, A_sum_axis0.shape
```

```{.python .input}
#@tab tensorflow
A_sum_axis0 = tf.reduce_sum(A, axis=0)
A_sum_axis0, A_sum_axis0.shape
```

Specifying `axis=1` will reduce the column dimension (axis 1) by summing up elements of all the columns.

```{.python .input}
A_sum_axis1 = A.sum(axis=1)
A_sum_axis1, A_sum_axis1.shape
```

```{.python .input}
#@tab pytorch
A_sum_axis1 = A.sum(axis=1)
A_sum_axis1, A_sum_axis1.shape
```

```{.python .input}
#@tab tensorflow
A_sum_axis1 = tf.reduce_sum(A, axis=1)
A_sum_axis1, A_sum_axis1.shape
```

Reducing a matrix along both rows and columns via summation
is equivalent to summing up all the elements of the matrix.

```{.python .input}
A.sum(axis=[0, 1]) == A.sum() # Same as `A.sum()`
```

```{.python .input}
#@tab pytorch
A.sum(axis=[0, 1]) == A.sum() # Same as `A.sum()`
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(A, axis=[0, 1])  tf.reduce_sum(A) # Same as `tf.reduce_sum(A)`
```

[**A related quantity is the *mean*, also called the *average*.**]
We calculate the mean by dividing the sum by the total number of elements.
Since computing the `mean` is a commonly used operation, there is a function for that. It works just like `sum`.

```{.python .input}
A.mean(), A.sum() / A.size
```

```{.python .input}
#@tab pytorch
A.mean(), A.sum() / A.numel()
```

```{.python .input}
#@tab tensorflow
tf.reduce_mean(A), tf.reduce_sum(A) / tf.size(A).numpy()
```

Likewise, the function for calculating the mean can also reduce a tensor along the specified axes.

```{.python .input}
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

```{.python .input}
#@tab pytorch
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

```{.python .input}
#@tab tensorflow
tf.reduce_mean(A, axis=0), tf.reduce_sum(A, axis=0) / A.shape[0]
```

## Non-Reduction Sum
:label:`subseq_lin-alg-non-reduction`

Sometimes it can be useful to [**keep the number of axes unchanged**]
when invoking the function for calculating the sum or mean. This matters for instance when we want to use the broadcast mechanism. 

```{.python .input}
sum_A = A.sum(axis=1, keepdims=True)
sum_A, sum_A.shape
```

```{.python .input}
#@tab pytorch
sum_A = A.sum(axis=1, keepdims=True)
sum_A, sum_A.shape
```

```{.python .input}
#@tab tensorflow
sum_A = tf.reduce_sum(A, axis=1, keepdims=True)
sum_A, sum_A.shape
```

For instance,
since `sum_A` still keeps its two axes after summing each row, we can (**divide `A` by `sum_A` with broadcasting**) to create a matrix where each row sums up to $1$.

```{.python .input}
A / sum_A
```

```{.python .input}
#@tab pytorch
A / sum_A
```

```{.python .input}
#@tab tensorflow
A / sum_A
```

If we want to calculate [**the cumulative sum of elements of `A` along some axis**], say `axis=0` (row by row),
we can call the `cumsum` function. By design, this function will not reduce the input tensor along any axis.

```{.python .input}
A.cumsum(axis=0)
```

```{.python .input}
#@tab pytorch
A.cumsum(axis=0)
```

```{.python .input}
#@tab tensorflow
tf.cumsum(A, axis=0)
```

## Dot Products

So far, we have only performed elementwise operations, sums, and averages. And if this was all we could do, linear algebra probably would not deserve its own section. One of the most fundamental operations is the dot product.
Given two vectors $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$, their *dot product* $\mathbf{x}^\top \mathbf{y}$ (or $\langle \mathbf{x}, \mathbf{y}  \rangle$) is a sum over the products of the elements at the same position: $\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{d} x_i y_i$.

[~~The *dot product* of two vectors is a sum over the products of the elements at the same position~~]

```{.python .input}
y = np.ones(3)
x, y, np.dot(x, y)
```

```{.python .input}
#@tab pytorch
y = torch.ones(3, dtype = torch.float32)
x, y, torch.dot(x, y)
```

```{.python .input}
#@tab tensorflow
y = tf.ones(3, dtype=tf.float32)
x, y, tf.tensordot(x, y, axes=1)
```

Note that
(**we can express the dot product of two vectors equivalently by performing an elementwise multiplication and then a sum:**)

```{.python .input}
np.sum(x * y)
```

```{.python .input}
#@tab pytorch
torch.sum(x * y)
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(x * y)
```

Dot products are useful in a wide range of contexts.
For example, given some set of values,
denoted by a vector $\mathbf{x}  \in \mathbb{R}^n$
and a set of weights denoted by $\mathbf{w} \in \mathbb{R}^n$,
the weighted sum of the values in $\mathbf{x}$
according to the weights $\mathbf{w}$
could be expressed as the dot product $\mathbf{x}^\top \mathbf{w}$.
When the weights are non-negative
and sum to one, i.e., $\left(\sum_{i=1}^{n} {w_i} = 1\right)$,
the dot product expresses a *weighted average*.
After normalizing two vectors to have the unit length,
the dot products express the cosine of the angle between them.
We will formally introduce this notion of *length* later in this section.


## Matrix-Vector Products

Now that we know how to calculate dot products,
we can begin to understand *matrix-vector products*.
Recall the matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$
and the vector $\mathbf{x} \in \mathbb{R}^n$
defined and visualized in :eqref:`eq_matrix_def` and :eqref:`eq_vec_def` respectively.
Let us start off by visualizing the matrix $\mathbf{A}$ in terms of its row vectors

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix},$$

where each $\mathbf{a}^\top_{i} \in \mathbb{R}^n$
is a row vector representing the $i^\mathrm{th}$ row of the matrix $\mathbf{A}$.

[**The matrix-vector product $\mathbf{A}\mathbf{x}$
is simply a column vector of length $m$,
whose $i^\mathrm{th}$ element is the dot product $\mathbf{a}^\top_i \mathbf{x}$:**]

$$
\mathbf{A}\mathbf{x}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix}\mathbf{x}
= \begin{bmatrix}
 \mathbf{a}^\top_{1} \mathbf{x}  \\
 \mathbf{a}^\top_{2} \mathbf{x} \\
\vdots\\
 \mathbf{a}^\top_{m} \mathbf{x}\\
\end{bmatrix}.
$$

We can think of multiplication by a matrix $\mathbf{A}\in \mathbb{R}^{m \times n}$
as a transformation that projects vectors
from $\mathbb{R}^{n}$ to $\mathbb{R}^{m}$.
These transformations turn out to be remarkably useful.
For example, we can represent rotations
as multiplications by a square matrix.
As we will see in subsequent chapters,
we can also use matrix-vector products
to describe the most intensive calculations
required when computing each layer in a neural network
given the values of the previous layer.

:begin_tab:`mxnet`
Expressing matrix-vector products in code with tensors,
we use the same `dot` function as for dot products.
When we call `np.dot(A, x)` with a matrix `A` and a vector `x`,
the matrix-vector product is performed.
Note that the column dimension of `A` (its length along axis 1)
must be the same as the dimension of `x` (its length).
:end_tab:

:begin_tab:`pytorch`
Expressing matrix-vector products in code with tensors, we use
the `mv` function. When we call `torch.mv(A, x)` with a matrix
`A` and a vector `x`, the matrix-vector product is performed.
Note that the column dimension of `A` (its length along axis 1)
must be the same as the dimension of `x` (its length). 
PyTorch has a convenience operator `@` that can be used in place of 
matrix-vector and also matrix-matrix products. In our context this means
that we can write `A@x` instead, thus greatly simplifying notation. 
:end_tab:

:begin_tab:`tensorflow`
Expressing matrix-vector products in code with tensors, we use
the `matvec` function. When we call `tf.linalg.matvec(A, x)` with a
matrix `A` and a vector `x`, the matrix-vector product is
performed. Note that the column dimension of `A` (its length along axis 1)
must be the same as the dimension of `x` (its length).
:end_tab:

```{.python .input}
A.shape, x.shape, np.dot(A, x)
```

```{.python .input}
#@tab pytorch
A.shape, x.shape, torch.mv(A, x), A@x
```

```{.python .input}
#@tab tensorflow
A.shape, x.shape, tf.linalg.matvec(A, x)
```

## Matrix-Matrix Multiplication

If you've gotten the hang of dot products and matrix-vector products,
then *matrix-matrix multiplication* should be straightforward.

Say that we have two matrices $\mathbf{A} \in \mathbb{R}^{n \times k}$ and $\mathbf{B} \in \mathbb{R}^{k \times m}$:

$$\mathbf{A}=\begin{bmatrix}
 a_{11} & a_{12} & \cdots & a_{1k} \\
 a_{21} & a_{22} & \cdots & a_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{n1} & a_{n2} & \cdots & a_{nk} \\
\end{bmatrix},\quad
\mathbf{B}=\begin{bmatrix}
 b_{11} & b_{12} & \cdots & b_{1m} \\
 b_{21} & b_{22} & \cdots & b_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
 b_{k1} & b_{k2} & \cdots & b_{km} \\
\end{bmatrix}.$$


Denote by $\mathbf{a}^\top_{i} \in \mathbb{R}^k$
the row vector representing the $i^\mathrm{th}$ row of the matrix $\mathbf{A}$,
and let $\mathbf{b}_{j} \in \mathbb{R}^k$
be the column vector from the $j^\mathrm{th}$ column of the matrix $\mathbf{B}$.
To produce the matrix product $\mathbf{C} = \mathbf{A}\mathbf{B}$, it is easiest to think of $\mathbf{A}$ in terms of its row vectors and $\mathbf{B}$ in terms of its column vectors:

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix},
\quad \mathbf{B}=\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}.
$$


Then the matrix product $\mathbf{C} \in \mathbb{R}^{n \times m}$ is produced as we simply compute each element $c_{ij}$ as the dot product $\mathbf{a}^\top_i \mathbf{b}_j$:

$$\mathbf{C} = \mathbf{AB} = \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix}
\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \mathbf{b}_1 & \mathbf{a}^\top_{1}\mathbf{b}_2& \cdots & \mathbf{a}^\top_{1} \mathbf{b}_m \\
 \mathbf{a}^\top_{2}\mathbf{b}_1 & \mathbf{a}^\top_{2} \mathbf{b}_2 & \cdots & \mathbf{a}^\top_{2} \mathbf{b}_m \\
 \vdots & \vdots & \ddots &\vdots\\
\mathbf{a}^\top_{n} \mathbf{b}_1 & \mathbf{a}^\top_{n}\mathbf{b}_2& \cdots& \mathbf{a}^\top_{n} \mathbf{b}_m
\end{bmatrix}.
$$

[**We can think of the matrix-matrix multiplication $\mathbf{AB}$ as simply performing $m$ matrix-vector products or $m \times n$ dot products and stitching the results together to form an $n \times m$ matrix.**]
In the following snippet, we perform matrix multiplication on `A` and `B`.
Here, `A` is a matrix with 5 rows and 4 columns,
and `B` is a matrix with 4 rows and 3 columns.
After multiplication, we obtain a matrix with 2 rows and 3 columns.

```{.python .input}
B = np.ones(shape=(3, 3))
np.dot(A, B)
```

```{.python .input}
#@tab pytorch
B = torch.ones(3, 3)
torch.mm(A, B), A@B
```

```{.python .input}
#@tab tensorflow
B = tf.ones((3, 3), tf.float32)
tf.matmul(A, B)
```

Matrix-matrix multiplication can be simply called *matrix multiplication*, and should not be confused with the Hadamard product.


## Norms
:label:`subsec_lin-algebra-norms`

Some of the most useful operators in linear algebra are *norms*.
Informally, the norm of a vector tells us how *big* a vector is. 
For instance, the $\ell_2$ norm in $\mathbb{R}^3$ measures the length of a vector in three dimensions. 
As such, the notion of *size* under consideration here
concerns not dimensionality
but rather the magnitude of the components. 

A norm is a function $\| \cdot \|$ that maps a vector
to a scalar, satisfying the following three properties:

1. Given any vector $\mathbf{x}$, if we scale (all elements of) the vector 
   by a scalar $\alpha \in \mathbb{R}$, its norm scales accordingly:
   $$\|\alpha \mathbf{x}\| = |\alpha| \|\mathbf{x}\|$$
2. For any vectors $\mathbf{x}$ and $\mathbf{y}$ norms satisfy the 
   triangle inequality
   $$\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|.$$
3. The norm of a vector is nonnegative and it only vanishes if the vector is zero. That is, 
   $$\|\mathbf{x}\| > 0 \text{ for all } \mathbf{x} \neq 0.$$

As such, norms encode different ways of measuring the length of a vector. The (regular) Euclidean norm amounts to the square root of the sum of squares of the vector elements. Hence we can write [**The $L_2$ *norm***] as

(**$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2}.$$**)

The method `norm` lets us compute it easily. 

```{.python .input}
u = np.array([3, -4])
np.linalg.norm(u)
```

```{.python .input}
#@tab pytorch
u = torch.tensor([3.0, -4.0])
torch.norm(u)
```

```{.python .input}
#@tab tensorflow
u = tf.constant([3.0, -4.0])
tf.norm(u)
```

Another popular norm is [**the $L_1$ *norm***]. The associated metric is also known as the Manhattan distance. The norm is defined as the sum of the absolute values of the vector elements:

(**$$\|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.$$**)

Compared to the $L_2$ norm, it is less sensitive to outliers, since large values do not get emphasized by squaring them. To compute the $L_1$ norm we compse the absolute value function with a sum over all elements. 

```{.python .input}
np.abs(u).sum()
```

```{.python .input}
#@tab pytorch
torch.abs(u).sum()
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(tf.abs(u))
```

Both the $L_2$ norm and the $L_1$ norm
are special cases of the more general $L_p$ *norm*:

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.$$

In the case of matrices, matters are more complicated. After all, matrices can be viewed both as collections of individual entries *and* as objects that operate on vectors and transform them into other vectors. For instance, we can ask by how much longer the matrix-vector product $\mathbf{X} \mathbf{v}$ could be relative to $\mathbf{v}$. This line of thought leads to a norm that equals the absolute value of the largest eigenvector of a matrix. For now we choose something that is a lot easier to compute:
[**the *Frobenius norm* of a matrix $\mathbf{X} \in \mathbb{R}^{m \times n}$**].
It is the square root of the sum of the squares of the matrix elements:

[**$$\|\mathbf{X}\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}.$$**]

It behaves as if it were an $L_2$ norm of a matrix-shaped vector.
Invoking the following function will calculate the Frobenius norm of a matrix.

```{.python .input}
np.linalg.norm(np.ones((4, 9)))
```

```{.python .input}
#@tab pytorch
torch.norm(torch.ones((4, 9)))
```

```{.python .input}
#@tab tensorflow
tf.norm(tf.ones((4, 9)))
```

While we do not want to get too far ahead of ourselves,
we can plant some intuition already about why these concepts are useful.
In deep learning, we are often trying to solve optimization problems:
*maximize* the probability assigned to observed data;
*maximize* the revenue associated with a recommender model; 
*minimize* the distance between predictions
and the ground-truth observations; 
*minimize* the distance between representations of photos of the same person while *maximizing* the distance between representations of photos of different persons. 
Oftentimes, the objectives of deep learning algorithms, which are arguably some of the most important components of a model, are expressed as norms. 


## Summary

In just this section,
we have taught you all the linear algebra
that you will need to understand
a remarkable chunk of modern deep learning.
There is a lot more to linear algebra
and a lot of that mathematics is useful for machine learning.
For example, matrices can be decomposed into factors,
and these decompositions can reveal
low-dimensional structure in real-world datasets.
There are entire subfields of machine learning
that focus on using matrix decompositions
and their generalizations to high-order tensors
to discover structure in datasets and solve prediction problems.
But this book focuses on deep learning.
And we believe you will be much more inclined to learn more mathematics
once you have gotten your hands dirty
deploying useful machine learning models on real datasets.
So while we reserve the right to introduce more mathematics much later on,
we will wrap up this section here.

If you are eager to learn more about linear algebra,
you may refer to either a number of excellent books and online resources :cite:`Strang.1993,Kolter.2008,Petersen.Pedersen.ea.2008` or read more in our
[appendix on linear algebra](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/geometry-linear-algebraic-ops.html).

For now, the following will suffice:
* Scalars, vectors, matrices, and tensors of are the basic mathematical objects used in linear algebra. In particular, vectors generalize scalars, matrices generalize vectors, and lastly, tensors generalize matrices. 
* Scalars, vectors, matrices, and tensors have zero, one, two, and an arbitrary number of axes, respectively.
* A tensor can be sliced or it can be reduced along the specified axes by operations such as `sum` and `mean`.
* Elementwise multiplication of two matrices is called their Hadamard product. It is different from matrix multiplication both in terms of effect (elementwise) and speed (quadratic rather than cubic time to compute).
* In deep learning, we often work with norms such as the $L_1$ norm, the $L_2$ norm, and the Frobenius norm.


## Exercises

1. Prove that the transpose of the transpose of a matrix is the matrix itself: $(\mathbf{A}^\top)^\top = \mathbf{A}$.
1. Given two matrices $\mathbf{A}$ and $\mathbf{B}$, show that sum and transposition commute: $\mathbf{A}^\top + \mathbf{B}^\top = (\mathbf{A} + \mathbf{B})^\top$.
1. Given any square matrix $\mathbf{A}$, is $\mathbf{A} + \mathbf{A}^\top$ always symmetric? Can you prove the result by using only the result of the previous two exercises?
1. We defined the tensor `X` of shape (2, 3, 4) in this section. What is the output of `len(X)`? Write your answer without implementing any code, then check your answer using code. 
1. For a tensor `X` of arbitrary shape, does `len(X)` always correspond to the length of a certain axis of `X`? What is that axis?
1. Run `A / A.sum(axis=1)` and see what happens. Can you analyze the reason?
1. When traveling between two points in downtown Manhattan, what is the distance that you need to cover in terms of the coordinates, i.e., in terms of avenues and streets? Can you travel diagonally?
1. Consider a tensor with shape (2, 3, 4). What are the shapes of the summation outputs along axis 0, 1, and 2?
1. Feed a tensor with 3 or more axes to the `linalg.norm` function and observe its output. What does this function compute for tensors of arbitrary shape?
1. Define three large matrices, say $\mathbf{A} \in \mathbb{R}^{2^{10} \times 2^{16}}$, $\mathbf{B} \in \mathbb{R}^{2^{16} \times 2^{5}}$ and $\mathbf{C} \in \mathbb{R}^{2^{5} \times 2^{14}}$, for instance initialized with Gaussian random variables. You want to compute the product $\mathbf{A} \mathbf{B} \mathbf{C}$. Is there any difference in memory footprint and speed, depending on whether you compute $(\mathbf{A} \mathbf{B}) \mathbf{C}$ or $\mathbf{A} (\mathbf{B} \mathbf{C})$. Why?
1. Define three large matrices, say $\mathbf{A} \in \mathbb{R}^{2^{10} \times 2^{16}}$, $\mathbf{B} \in \mathbb{R}^{2^{16} \times 2^{5}}$ and $\mathbf{C} \in \mathbb{R}^{2^{5} \times 2^{16}}$. Is there any difference in speed depending on whether you compute $\mathbf{A} \mathbf{B}$ or $\mathbf{A} \mathbf{C}^\top$? Why? What changes if you initialize $\mathbf{C} = \mathbf{B}^\top$ without cloning memory? Why?
1. Define three matrices, say $\mathbf{A}, \mathbf{B}, \mathbf{C} \in \mathbb{R}^{100 \times 200}$. Constitute a tensor with 3 axes by stacking $[\mathbf{A}, \mathbf{B}, \mathbf{C}]$. What is the dimensionality? Slice out the second coordinate of the third axis to recover $\mathbf{B}$. Check that your answer is correct.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/30)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/31)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/196)
:end_tab:

```{.python .input}

```
