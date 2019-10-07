# Linear Algebra
:label:`chapter_linear_algebra`

Now that you can store and manipulate data,
let us briefly review the subset of basic linear algebra
that you will need to understand and implement
most of models covered in this book.
Below, we introduce the basic concepts,
expressing each both through mathematical notation
and the corresponding implementation in code.
If you are already confident in your basic linear algebra,
feel free to skim or skip this section.

```{.python .input  n=1}
from mxnet import np, npx
npx.set_np()
```

## Scalars

If you never studied linear algebra or machine learning,
then your past experience with math probably consisted
of thinking about one number at a time.
And, if you ever balanced a checkbook
or even paid for dinner at a restaurant
then you already know how to do basic things
like adding and multiplying pairs of numbers.
For example, the temperature in Palo Alto is $52$ degrees Fahrenheit.
Formally, we call values consisting
of just one numerical quantity *scalars*.
If you wanted to convert this value to Celsius
(the metric system's more sensible temperature scale),
you would evaluate the expression $c = \frac{5}{9}(f - 32)$, setting $f$ to $52$.
In this equation, each of the terms---$5$, $9$, and $32$---are scalar values.
The placeholders $c$ and $f$ are called *variables*
and they represented unknown scalar values.

In this book, we adopt the mathematical notation
where scalar variables are denoted
by ordinary lower-cased letters (e.g., $x$, $y$, and $z$).
We denote the space of all (continuous) *real-valued* scalars by $\mathbb{R}$.
For expedience, we will punt on rigorous definitions
of what precisely *space* is,
but just remember for now that the expression $x \in \mathbb{R}$
is a formal way to say that $x$ is a real-valued scalar.
The symbol $\in$ can be pronounced "in"
and simply denotes membership in a set.
Analogously, we could write $x,y \in \{0,1\}$
to state that $x$ and $y$ are numbers
whose value can only be $0$ or $1$.

In MXNet code, a scalar is represented by an `ndarray` with just one element.
In the next snippet, we instantiate two scalars
and perform some familiar arithmetic operations with them,
namely addition, multiplication, division, and exponentiation.

```{.python .input  n=2}
x = np.array(3.0)
y = np.array(2.0)

x + y, x * y, x / y, x ** y
```

## Vectors

You can think of a vector as simply a list of scalar values.
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
In math notation, we will usually denote vectors as bold-faced,
lower-cased letters (e.g., $\mathbf{x}$, $\mathbf{y}$, and $\mathbf{z})$.

In MXNet, we work with vectors via $1$-dimensional `ndarray`s.
In general `ndarray`s can have arbitrary lengths,
subject to the memory limits of your machine.

```{.python .input  n=3}
x = np.arange(4)
x
```

We can refer to any element of a vector by using a subscript.
For example, we can refer to the $i^\mathrm{th}$ element of $\mathbf{x}$ by $x_i$.
Note that the element $x_i$ is a scalar,
so we do not bold-face the font when referring to it.
Extensive literature considers column vectors to be the default
orientation of vectors, so does this book.
In math, a vector $\mathbf{x}$ can be written as

$$
\mathbf{x} =
\begin{bmatrix}
    x_{1}  \\
    x_{2}  \\
    \vdots  \\
    x_{n}
\end{bmatrix},
$$

where $x_1, \ldots, x_n$ are elements of the vector.
In code, we access any element by indexing into the `ndarray`.

```{.python .input  n=4}
x[3]
```

### Length, Dimensionality, and Shape

Let us revisit some concepts from :numref:`chapter_ndarray`.
A vector is just an array of numbers.
And just as every array has a length, so does every vector.
In math notation, if we want to say that a vector $\mathbf{x}$
consists of $n$ real-valued scalars,
we can express this as $\mathbf{x} \in \mathbb{R}^n$.
The length of a vector is commonly called the *dimension* of the vector.

As with an ordinary Python array, we can access the length of an `ndarray`
by calling Python's built-in `len()` function.

```{.python .input  n=5}
len(x)
```

When an `ndarray` represents a vector (with precisely one axis),
we can also access its length via the `.shape` attribute.
The shape is a tuple that lists the length (dimensionality)
along each axis of the `ndarray`.
For `ndarray`s with just one axis, the shape has just one element.

```{.python .input  n=6}
x.shape
```

Note that the word "dimension" tends to get overloaded
in these contexts and this tends to confuse people.
To clarify, we use the dimensionality of a *vector* or an *axis*
to refer to its length, i.e., the number of elements of a vector or an axis.
However, we use the dimensionality of an `ndarray`
to refer to the number of axes that an `ndarray` has.
In this sense, the dimensionality of an `ndarray`'s some axis
will be the length of that axis.


## Matrices

Just as vectors generalize scalars from order $0$ to order $1$,
matrices generalize vectors from order $1$ to order $2$.
Matrices, which we will typically denote with bold-faced, capital letters
(e.g., $\mathbf{X}$, $\mathbf{Y}$, and $\mathbf{Z}$),
are represented in code as `ndarray`s with $2$ axes.

In math notation, we use $\mathbf{A} \in \mathbb{R}^{n \times m}$
to express that the matrix $\mathbf{A}$ consists of $n$ rows and $m$ columns of real-valued scalars.
Visually, we can illustrate any matrix $\mathbf{A} \in \mathbb{R}^{n \times m}$ as a table,
where each element $a_{ij}$ belongs to the $i^{\text{th}}$ row and $j^{\text{th}}$ column:

$$\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1m} \\ a_{21} & a_{22} & \cdots & a_{2m} \\ \vdots & \vdots & \ddots & \vdots \\ a_{n1} & a_{n2} & \cdots & a_{nm} \\ \end{bmatrix}.$$
:eqlabel:`eq_matrix_def`


For any $\mathbf{A} \in \mathbb{R}^{n \times m}$, the shape of $\mathbf{A}$
is ($n$, $m$) or $n \times m$.
We can create an $n \times m$ matrix in MXNet
by specifying a shape with two components $n$ and $m$
when calling any of our favorite functions for instantiating an `ndarray`.

```{.python .input  n=24}
A = np.arange(20).reshape(5, 4)
A
```

We can access the scalar element $a_{ij}$ of a matrix $\mathbf{A}$ in :eqref:`eq_matrix_def`
by specifying the indices for the row ($i$) and column ($j$) respectively,
such as $[\mathbf{A}]_{ij}$.
To keep notation simple, commas are inserted to separate indices only when necessary,
such as $a_{2,3j}$ and $[\mathbf{A}]_{2i-1,3}$.


Sometimes, we want to flip the axes.
When we exchange a matrix's rows and columns,
the result is called the *transpose* of the matrix.
Formally, we signify a matrix $\mathbf{A}$'s transpose by $\mathbf{A}^\top$
and if $B = A^\top$, then $b_{ij} = a_{ji}$ for any $i$ and $j$.
Thus, the transpose of $\mathbf{A}$ in :eqref:`eq_matrix_def` is
a $m \times n$ matrix:

$$
\mathbf{A}^\top =
\begin{bmatrix}
    a_{11} & a_{21} & \dots  & a_{n1} \\
    a_{12} & a_{22} & \dots  & a_{n2} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{1m} & a_{2m} & \dots  & a_{nm}
\end{bmatrix}.
$$



In code, we access a matrix's transpose via the `.T` attribute.

```{.python .input  n=8}
print(A.T)
```

Matrices are useful data structures: 
they allow us to organize data that have different modalities of variation. 
For example, rows in our matrix might correspond to different houses (data points), 
while columns might correspond to different attributes.
This should sound familiar if you have ever used spreadsheet software or
have read :ref:`subsec_data_preprocessing`.
Thus, although the default orientation of a single vector is a column vector,
in a matrix that represents a tabular dataset,
it is more conventional to treat each data point as a row vector in the matrix.
And, as we will see in later chapters,
this convention will enable common deep learning practices.
For example, along the outermost axis of an `ndarray`,
we can access or enumerate mini-batches of data points,
or just data points if no mini-batch exists.


## Tensors

Just as vectors generalize scalars, and matrices generalize vectors, we can build data structures with even more axes. Tensors give us a generic way of describing `ndarray`s with an arbitrary number of axes. Vectors, for example, are first-order tensors, and matrices are second-order tensors.
Tensors are denoted with capital letters 


Matrices, which we will typically denote with bold-faced, capital letters
(e.g., $\mathbf{X}$, $\mathbf{Y}$, and $\mathbf{Z}$),
are represented in code as `ndarray`s with $2$ axes.


Tensors will become more important when we start working with images, which arrive as `ndarray`s with 3 axes corresponding to the height, width, and a *channel* axis for stacking the color channels (red, green, and blue). For now, we will skip over higher order tensors and focus on the basics.

```{.python .input  n=9}
X = np.arange(24).reshape(2, 3, 4)
X
```

## Basic properties of tensor arithmetic

Scalars, vectors, matrices, and tensors of any order
have some nice properties that often come in handy.
For example, you might have noticed
from the definition of an element-wise operation,
any elementwise unary operation does not change the shape of its operand.
Similarly, given any two tensors with the same shape,
the result of any binary element-wise operation
will be a tensor of that same shape.
The same holds for multiplication by a scalar.
Using math notation, given any two tensors $X, Y \in \mathcal{R}^{m \times n}$,
$\alpha X + Y \in  \mathcal{R}^{m \times n}$
(numerical mathematicians call this the AXPY operation).

```{.python .input  n=10}
a = 2
x = np.ones(3)
y = np.zeros(3)
print(x.shape)
print(y.shape)
print((a * x).shape)
print((a * x + y).shape)
```

Shape is not the the only property preserved
under addition and multiplication by a scalar.
These operations also preserve membership in a vector space.
But, again we will punt discussion of *vector spaces*
in favor of information more critical to getting your first models up and running.

## Sums and means

One useful operation that we can perform with arbitrary tensors
is to calculate the sum of their elements.
In mathematical notation, we express sums using the $\sum$ symbol.
To express the sum of the elements in a vector $\mathbf{u}$ of length $d$,
we write $\sum_{i=1}^d u_i$. In code, we can just call ``sum()``.

```{.python .input  n=11}
print(x)
print(x.sum())
```

We can express sums over the elements of tensors of arbitrary shape.
For example, the sum of the elements of an $m \times n$ matrix $A$ could be written $\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$.

```{.python .input  n=12}
print(A)
print(A.sum())
```

A related quantity is the *mean*, which is also called the *average*.
We calculate the mean by dividing the sum by the total number of elements.
With mathematical notation, we could write the average
over a vector $\mathbf{u}$ as $\frac{1}{d} \sum_{i=1}^{d} u_i$
and the average over a matrix $A$ as  $\frac{1}{n \cdot m} \sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$.
In code, we could just call ``mean()`` on tensors of arbitrary shape:

```{.python .input  n=13}
print(A.mean())
print(A.sum() / A.size)
```

## Dot products

So far, we have only performed element-wise operations, sums and averages. And if this was all we could do, linear algebra probably would not deserve its own chapter. However, one of the most fundamental operations is the dot product. Given two vectors $\mathbf{u}$ and $\mathbf{v}$, the dot product $\mathbf{u}^T \mathbf{v}$ is a sum over the products of the corresponding elements: $\mathbf{u}^T \mathbf{v} = \sum_{i=1}^{d} u_i \cdot v_i$.

```{.python .input  n=14}
x = np.arange(4)
y = np.ones(4)
print(x, y, np.dot(x, y))
```

Note that we can express the dot product of two vectors `dot(x, y)` equivalently by performing an element-wise multiplication and then a sum:

```{.python .input  n=15}
np.sum(x * y)
np.dot(x, y)
```

Dot products are useful in a wide range of contexts.
For example, given some set of values,
denoted by a vector $\mathbf{u}$
and a set of weights denoted by $\mathbf{w}$,
the weighted sum of the values in $\mathbf{u}$
according to the weights $\mathbf{w}$
could be expressed as the dot product $\mathbf{u}^T \mathbf{w}$.
When the weights are non-negative
and sum to one (i.e., $\left(\sum_{i=1}^{d} {w_i} = 1\right)$),
the dot product expresses a *weighted average*.
After normalizing two vectors to have length one,
the dot products express the cosine of the angle between them.
We formally introduce this notion of *length* below in the section on norms.


## Matrix-vector products

Now that we know how to calculate dot products,
we can begin to understand matrix-vector products.
Let's start off by visualizing a matrix $A$ and a column vector $\mathbf{x}$.

$$A=\begin{bmatrix}
 a_{11} & a_{12} & \cdots & a_{1m} \\
 a_{21} & a_{22} & \cdots & a_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{n1} & a_{n2} & \cdots & a_{nm} \\
\end{bmatrix},\quad\mathbf{x}=\begin{bmatrix}
 x_{1}  \\
 x_{2} \\
\vdots\\
 x_{m}\\
\end{bmatrix} $$

We can visualize the matrix in terms of its row vectors

$$A=
\begin{bmatrix}
\mathbf{a}^T_{1} \\
\mathbf{a}^T_{2} \\
\vdots \\
\mathbf{a}^T_n \\
\end{bmatrix},$$

where each $\mathbf{a}^T_{i} \in \mathbb{R}^{m}$
is a row vector representing the $i$-th row of the matrix $A$.

The matrix vector product $\mathbf{y} = A\mathbf{x}$
is simply a column vector $\mathbf{y} \in \mathbb{R}^n$,
where each entry $y_i$ is the dot product $\mathbf{a}^T_i \mathbf{x}$.

$$A\mathbf{x}=
\begin{bmatrix}
\mathbf{a}^T_{1}  \\
\mathbf{a}^T_{2}  \\
 \vdots  \\
\mathbf{a}^T_n \\
\end{bmatrix}
\begin{bmatrix}
 x_{1}  \\
 x_{2} \\
\vdots\\
 x_{m}\\
\end{bmatrix}
= \begin{bmatrix}
 \mathbf{a}^T_{1} \mathbf{x}  \\
 \mathbf{a}^T_{2} \mathbf{x} \\
\vdots\\
 \mathbf{a}^T_{n} \mathbf{x}\\
\end{bmatrix}
$$

You can think of multiplication by a matrix $A\in \mathbb{R}^{n \times m}$
as a transformation that projects vectors
from $\mathbb{R}^{m}$ to $\mathbb{R}^{n}$.

These transformations turn out to be remarkably useful.
For example, we can represent rotations
as multiplications by a square matrix.
As we will see in subsequent chapters,
we can also use matrix-vector products
to describe the most intensive calculations
required when computing each layer in a neural network
given the values of the previous layer.

Expressing matrix-vector products in code with `ndarray`,
we use the same ``dot()`` function as for dot products.
When we call ``np.dot(A, x)`` with a matrix ``A`` and a vector ``x``,
MXNet knows to perform a matrix-vector product.
Note that the column dimension of ``A`` (its length along axis 1)
must be the same as the dimension of ``x`` (its length).

```{.python .input  n=16}
np.dot(A, x)
```

## Matrix-matrix multiplication

If you have gotten the hang of dot products and matrix-vector multiplication,
then matrix-matrix multiplications should be straightforward.

Say that we have two matrices, $A \in \mathbb{R}^{n \times k}$ and $B \in \mathbb{R}^{k \times m}$:

$$A=\begin{bmatrix}
 a_{11} & a_{12} & \cdots & a_{1k} \\
 a_{21} & a_{22} & \cdots & a_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{n1} & a_{n2} & \cdots & a_{nk} \\
\end{bmatrix},\quad
B=\begin{bmatrix}
 b_{11} & b_{12} & \cdots & b_{1m} \\
 b_{21} & b_{22} & \cdots & b_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
 b_{k1} & b_{k2} & \cdots & b_{km} \\
\end{bmatrix}$$

To produce the matrix product $C = AB$, it's easiest to think of $A$ in terms of its row vectors and $B$ in terms of its column vectors:

$$A=
\begin{bmatrix}
\mathbf{a}^T_{1} \\
\mathbf{a}^T_{2} \\
\vdots \\
\mathbf{a}^T_n \\
\end{bmatrix},
\quad B=\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}.
$$

Note here that each row vector $\mathbf{a}^T_{i}$ lies in $\mathbb{R}^k$ and that each column vector $\mathbf{b}_j$ also lies in $\mathbb{R}^k$.

Then to produce the matrix product $C \in \mathbb{R}^{n \times m}$ we simply compute each entry $c_{ij}$ as the dot product $\mathbf{a}^T_i \mathbf{b}_j$.

$$C = AB = \begin{bmatrix}
\mathbf{a}^T_{1} \\
\mathbf{a}^T_{2} \\
\vdots \\
\mathbf{a}^T_n \\
\end{bmatrix}
\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}
= \begin{bmatrix}
\mathbf{a}^T_{1} \mathbf{b}_1 & \mathbf{a}^T_{1}\mathbf{b}_2& \cdots & \mathbf{a}^T_{1} \mathbf{b}_m \\
 \mathbf{a}^T_{2}\mathbf{b}_1 & \mathbf{a}^T_{2} \mathbf{b}_2 & \cdots & \mathbf{a}^T_{2} \mathbf{b}_m \\
 \vdots & \vdots & \ddots &\vdots\\
\mathbf{a}^T_{n} \mathbf{b}_1 & \mathbf{a}^T_{n}\mathbf{b}_2& \cdots& \mathbf{a}^T_{n} \mathbf{b}_m
\end{bmatrix}
$$

You can think of the matrix-matrix multiplication $AB$ as simply performing $m$ matrix-vector products and stitching the results together to form an $n \times m$ matrix. Just as with ordinary dot products and matrix-vector products, we can compute matrix-matrix products in MXNet by using ``dot()``.

```{.python .input  n=17}
B = np.ones(shape=(4, 3))
np.dot(A, B)
```

## Norms

Before we can start implementing models,
there is one last concept we are going to introduce.
Some of the most useful operators in linear algebra are norms.
Informally, they tell us how *big* a vector or matrix is.
The notion of *size* under consideration here
concerns not dimensionality
but rather the magnitude of the components.
We represent norms with the notation $\|\cdot\|$.
The $\cdot$ in this expression is just a placeholder.
For example, we would represent the norm of a vector $\mathbf{x}$
or matrix $A$ as $\|\mathbf{x}\|$ or $\|A\|$, respectively.

All norms must satisfy a handful of properties:

1. $\|\alpha A\| = |\alpha| \|A\|$
1. $\|A + B\| \leq \|A\| + \|B\|$
1. $\|A\| \geq 0$
1. If $\forall {i,j}, a_{ij} = 0$, then $\|A\|=0$

To put it in words, the first rule says
that if we scale all the components of a matrix or vector
by a constant factor $\alpha$,
its norm also scales by the *absolute value*
of the same constant factor.
The second rule is the familiar triangle inequality.
The third rule simply says that the norm must be non-negative.
That makes sense, in most contexts the smallest *size* for anything is 0.
The final rule requires that the smallest norm is achieved
by a matrix or vector consisting of all zeros.
It is possible to define a norm that gives zero norm to nonzero matrices,
but you cannot give nonzero norm to zero matrices.
That may seem like a mouthful, but if you digest it
then you probably have digested the important concepts here.

You might notice that norms sound a lot like measures of distance.
And you remember Euclidean distances
(think Pythagoras' theorem) from grade school,
then the concepts of non-negativity and the triangle inequality might ring a bell.

In fact, the Euclidean distance $\sqrt{x_1^2 + \cdots + x_n^2}$ is a norm.
Specifically it is the $\ell_2$-norm.
We call the analogous computation,
performed over the entries of a matrix:
$\sqrt{\sum_{i,j} a_{ij}^2}$,
the *Frobenius norm*.
In machine learning, we work more often
with the squared $\ell_2$ norm (notated $\ell_2^2$).
You will also frequently encounter the $\ell_1$ norm,
which is expressed as the sum of the absolute values of the components.
As compared to the $\ell_2$ norm,
it is less influenced by outliers.

In code, we can calculate the $\ell_2$ norm of an `ndarray` by calling ``norm()``.

```{.python .input  n=18}
np.linalg.norm(x)
```

To calculate the L1-norm, we compose
the absolute value function with a sum over the elements.

```{.python .input  n=19}
np.abs(x).sum()
```

### Norms and objectives

While we do not want to get too far ahead of ourselves,
we can plant some intuition already about why these concepts are useful.
In machine learning, we are often trying to solve optimization problems:
*Maximize* the probability assigned to observed data.
*Minimize* the distance between predictions
and the ground-truth observations.
Assign vector representations to items (like words, products, or news articles)
such that the distance between similar items is minimized,
and the distance between dissimilar items is maximized.
Oftentimes, the objectives, perhaps the most important components
of machine learning algorithms (besides the data),
are expressed as norms.



## Summary

In just a few pages (or one Jupyter notebook),
we have taught you all the linear algebra
that you will need to understand
a remarkable chunk of modern deep learning.
There is a *lot* more to linear algebra
and a lot of that math *is* useful for machine learning.
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
So while we reserve the right to introduce more math much later on,
we will wrap up this chapter here.

If you are eager to learn more about linear algebra,
here are some of our favorite resources on the topic

* For a solid primer on basics, check out Gilbert Strang's book [Introduction to Linear Algebra](http://math.mit.edu/~gs/linearalgebra/)
* Zico Kolter's [Linear Algebra Review and Reference](http://www.cs.cmu.edu/~zkolter/course/15-884/linalg-review.pdf)
* Kaare Brandt Peterson and Michael Syskind Peterson's [Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2317)

![](../img/qr_linear-algebra.svg)
