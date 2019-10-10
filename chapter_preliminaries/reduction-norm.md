# Reduction, Multiplication, and Norms
:label:`sec_reduction-norm`

We have introduced scalars, vectors, matrices, and tensors in :numref:`sec_scalar-tensor`.
Without operations on such basic mathematical objects in linear algebra,
we cannot design any meaningful procedure for deep learning.
Specifically, as we will see in subsequent chapters,
reduction, multiplication, and norms are arguably
among the most commonly used operations.
In this section, we start off by reviewing and summarizing basic properties of tensor arithmetic,
then illustrate those operations.


## Basic Properties of Tensor Arithmetic

Scalars, vectors, matrices, and tensors of an arbitrary number of axes
have some nice properties that often come in handy.
For example, you might have noticed
from the definition of an elementwise operation
that any elementwise unary operation does not change the shape of its operand.
Similarly, given any two tensors with the same shape,
the result of any binary elementwise operation
will be a tensor of that same shape.
For example, adding two matrices of the same shape
performs elementwise addition over these two matrices.

```{.python .input}
from mxnet import np, npx
npx.set_np()

A = np.arange(20).reshape(5, 4)
B = np.arange(20).reshape(5, 4)
A + B
```

Specifically, elementwise multiplication of two matrices is called their *Hadamard product* (math notation $\odot$).
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

Multiplying or adding a tensor by a scalar also does not change the shape of the tensor,
where each element of the operand tensor will be added or multiplied by the scalar.

```{.python .input}
a = 2
X = np.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

## Reduction

One useful operation that we can perform with arbitrary tensors
is to calculate the sum of their elements.
In mathematical notation, we express sums using the $\sum$ symbol.
To express the sum of the elements in a vector $\mathbf{x}$ of length $d$,
we write $\sum_{i=1}^d x_i$. In code, we can just call the `sum` function.

```{.python .input  n=11}
x = np.arange(4)
x, x.sum()
```

We can express sums over the elements of tensors of arbitrary shape.
For example, the sum of the elements of an $m \times n$ matrix $\mathbf{A}$ could be written $\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$.

```{.python .input  n=12}
A.shape, A.sum()
```

By default, invoking the `sum` function *reduces* a tensor along all its axes to a scalar.
We can also specify the axes along which the tensor is reduced via summation.
Take matrices as an example. 
To reduce the row dimension (axis $0$) by summing up elements of all the rows,
we specify `axis=0` when invoking `sum`. 
Since the input matrix reduces along axis $0$ to generate the output vector,
the dimension of axis $0$ of the input is lost in the output shape.

```{.python .input}
A_sum_axis0 = A.sum(axis=0)
A_sum_axis0, A_sum_axis0.shape
```

Specifying `axis=1` will reduce the column dimension (axis $1$) by summing up elements of all the columns.
Thus, the dimension of axis $1$ of the input is lost in the output shape.

```{.python .input}
A_sum_axis1 = A.sum(axis=1)
A_sum_axis1, A_sum_axis1.shape
```

Reducing a matrix along both rows and columns via summation
is equivalent to summing up all the elements of the matrix.

```{.python .input}
A.sum(axis=[0, 1])  # Same as A.sum()
```

A related quantity is the *mean*, which is also called the *average*.
We calculate the mean by dividing the sum by the total number of elements.
In code, we could just call `mean` on tensors of arbitrary shape.

```{.python .input  n=13}
A.mean(), A.sum() / A.size
```

Like `sum`, `mean` can also reduce a tensor along the specified axes.

```{.python .input}
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

## Dot Products

So far, we have only performed elementwise operations, sums, and averages. And if this was all we could do, linear algebra probably would not deserve its own section. However, one of the most fundamental operations is the dot product. Given two vectors $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$, their *dot product* $\mathbf{x}^\top \mathbf{y}$ (or $\langle \mathbf{x}, \mathbf{y}  \rangle$) is a sum over the products of the elements at the same position: $\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{d} x_i y_i$.

```{.python .input  n=14}
y = np.ones(4)
x, y, np.dot(x, y)
```

Note that we can express the dot product of two vectors equivalently by performing an elementwise multiplication and then a sum:

```{.python .input  n=15}
np.sum(x * y)
```

Dot products are useful in a wide range of contexts.
For example, given some set of values,
denoted by a vector $\mathbf{x}  \in \mathbb{R}^d$
and a set of weights denoted by $\mathbf{w} \in \mathbb{R}^d$,
the weighted sum of the values in $\mathbf{x}$
according to the weights $\mathbf{w}$
could be expressed as the dot product $\mathbf{x}^\top \mathbf{w}$.
When the weights are non-negative
and sum to one (i.e., $\left(\sum_{i=1}^{d} {w_i} = 1\right)$),
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
The matrix-vector product $\mathbf{A}\mathbf{x}$
is simply a column vector of length $m$,
whose $i^\mathrm{th}$ element is the dot product $\mathbf{a}^\top_i \mathbf{x}$:

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

Expressing matrix-vector products in code with `ndarray`s,
we use the same `dot` function as for dot products.
When we call `np.dot(A, x)` with a matrix `A` and a vector `x`,
the matrix-vector product is performed.
Note that the column dimension of `A` (its length along axis $1$)
must be the same as the dimension of `x` (its length).

```{.python .input  n=16}
A.shape, x.shape, np.dot(A, x)
```

## Matrix-Matrix Multiplication

If you have gotten the hang of dot products and matrix-vector products,
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

We can think of the matrix-matrix multiplication $\mathbf{AB}$ as simply performing $m$ matrix-vector products and stitching the results together to form an $n \times m$ matrix. Just as with ordinary dot products and matrix-vector products, we can compute matrix-matrix multiplication by using the `dot` function.
In the following snippet, we perform matrix multiplication on `A` and `B`.
Here, `A` is a matrix with $5$ rows and $4$ columns,
and `B` is a matrix with $4$ rows and $3$ columns.
After multiplication, we obtain a matrix with $5$ rows and $3$ columns.

```{.python .input  n=17}
B = np.ones(shape=(4, 3))
np.dot(A, B)
```

Matrix-matrix multiplication can be simply called *matrix multiplication*, and should not be confused with the Hadamard product.


## Norms

Some of the most useful operators in linear algebra are *norms*.
Informally, the norm of a vector tells us how *big* a vector is.
The notion of *size* under consideration here
concerns not dimensionality
but rather the magnitude of the components.

In linear algebra, a vector norm is a function $f$ that maps a vector
to a scalar, satisfying a handful of properties.
Given any vector $\mathbf{x}$,
the first property says
that if we scale all the elements of a vector
by a constant factor $\alpha$,
its norm also scales by the *absolute value*
of the same constant factor:

$$f(\alpha \mathbf{x}) = |\alpha| f(\mathbf{x}).$$


The second property is the familiar triangle inequality:

$$f(\mathbf{x} + \mathbf{y}) \leq f(\mathbf{x}) + f(\mathbf{y}).$$


The third property simply says that the norm must be non-negative:

$$f(\mathbf{x}) \geq 0.$$

That makes sense, as in most contexts the smallest *size* for anything is 0.
The final property requires that the smallest norm is achieved and only achieved
by a vector consisting of all zeros.

$$\forall i, [\mathbf{x}]_i = 0 \Leftrightarrow f(\mathbf{x})=0.$$

You might notice that norms sound a lot like measures of distance.
And if you remember Euclidean distances
(think Pythagoras' theorem) from grade school,
then the concepts of non-negativity and the triangle inequality might ring a bell.
In fact, the Euclidean distance is a norm:
specifically it is the $\ell_2$ norm. 
Suppose that the elements in the $n$-dimensional vector
$\mathbf{x}$ are $x_1, \ldots, x_n$. 
The *$\ell_2$ norm* of $\mathbf{x}$ is the square root of the sum of the squares of the vector elements:

$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2},$$

where the subscript $2$ is often omitted in $\ell_2$ norms, i.e., $\|\mathbf{x}\|$ is equivalent to $\|\mathbf{x}\|_2$. In code, we can calculate the $\ell_2$ norm of a vector by calling `linalg.norm`.

```{.python .input  n=18}
u = np.array([3, -4])
np.linalg.norm(u)
```

In deep learning, we work more often
with the squared $\ell_2$ norm.
You will also frequently encounter the $\ell_1$ norm,
which is expressed as the sum of the absolute values of the vector elements:

$$\|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.$$

As compared with the $\ell_2$ norm,
it is less influenced by outliers.
To calculate the $\ell_1$ norm, we compose
the absolute value function with a sum over the elements.

```{.python .input  n=19}
np.abs(u).sum()
```

Both the $\ell_2$ norm and the $\ell_1$ norm
are special cases of the more general $\ell_p$ norm:

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.$$

Analogous to $\ell_2$ norms of vectors,
the *Frobenius norm* of a matrix $\mathbf{X} \in \mathbb{R}^{m \times n}$
is the square root of the sum of the squares of the matrix elements:

$$\|\mathbf{X}\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}.$$

The Frobenius norm satisfies all the properties of vector norms.
It behaves as if it were an $\ell_2$ norm of a matrix-shaped vector. Invoking `linalg.norm` will calculate the Frobenius norm of a matrix.

```{.python .input}
np.linalg.norm(np.ones((4, 9)))
```

### Norms and Objectives

While we do not want to get too far ahead of ourselves,
we can plant some intuition already about why these concepts are useful.
In deep learning, we are often trying to solve optimization problems:
*maximize* the probability assigned to observed data;
*minimize* the distance between predictions
and the ground-truth observations.
Assign vector representations to items (like words, products, or news articles)
such that the distance between similar items is minimized,
and the distance between dissimilar items is maximized.
Oftentimes, the objectives, perhaps the most important components
of deep learning algorithms (besides the data),
are expressed as norms.



## More on Linear Algebra

In just :numref:`sec_scalar-tensor` and this section,
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
you may refer to either :numref:`chap_appendix_math`
or other excellent resources :cite:`Strang.Strang.Strang.ea.1993`:cite:`Kolter.2008`:cite:`Petersen.Pedersen.ea.2008`.


## Summary

* A tensor can be reduced along the specified axes by `sum` and `mean`.
* Elementwise multiplication of two matrices is called their Hadamard product. It is different from matrix multiplication.
* In deep learning, we often work with norms such as the $\ell_1$ norm, the $\ell_2$ norm, and the Frobenius norm.
* We can perform a variety of operations over scalars, vectors, matrices, and tensors with `ndarray` functions.


## Exercises

1. Consider a tensor with shape ($2$, $3$, $4$). What are the shapes of the summation outputs along axis $0$, $1$, and $2$?
1. Feed a tensor with 3 or more axes to the `linalg.norm` function and observe its output. What does this function compute for `ndarray`s of arbitrary shape?


## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/4974)

![](../img/qr_reduction-norm.svg)
