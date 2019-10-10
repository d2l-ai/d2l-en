# Scalars, Vectors, Matrices, and Tensors
:label:`sec_scalar-tensor`

Now that you can store and manipulate data,
let us briefly review the subset of basic linear algebra
that you will need to understand and implement
most of models covered in this book.
Below, we introduce the basic mathematical objects in linear algebra,
expressing each both through mathematical notation
and the corresponding implementation in code.

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

```{.python .input  n=1}
from mxnet import np, npx
npx.set_np()

x = np.array(3.0)
y = np.array(2.0)

x + y, x * y, x / y, x ** y
```

```{.json .output n=1}
[
 {
  "data": {
   "text/plain": "(array(5.), array(6.), array(1.5), array(9.))"
  },
  "execution_count": 1,
  "metadata": {},
  "output_type": "execute_result"
 }
]
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

```{.python .input  n=2}
x = np.arange(4)
x
```

```{.json .output n=2}
[
 {
  "data": {
   "text/plain": "array([0., 1., 2., 3.])"
  },
  "execution_count": 2,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

We can refer to any element of a vector by using a subscript.
For example, we can refer to the $i^\mathrm{th}$ element of $\mathbf{x}$ by $x_i$.
Note that the element $x_i$ is a scalar,
so we do not bold-face the font when referring to it.
Extensive literature considers column vectors to be the default
orientation of vectors, so does this book.
In math, a vector $\mathbf{x}$ can be written as

$$\mathbf{x} =\begin{bmatrix}x_{1}  \\x_{2}  \\ \vdots  \\x_{n}\end{bmatrix},$$
:eqlabel:`eq_vec_def`


where $x_1, \ldots, x_n$ are elements of the vector.
In code, we access any element by indexing into the `ndarray`.

```{.python .input  n=3}
x[3]
```

```{.json .output n=3}
[
 {
  "data": {
   "text/plain": "array(3.)"
  },
  "execution_count": 3,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

### Length, Dimensionality, and Shape

Let us revisit some concepts from :numref:`sec_ndarray`.
A vector is just an array of numbers.
And just as every array has a length, so does every vector.
In math notation, if we want to say that a vector $\mathbf{x}$
consists of $n$ real-valued scalars,
we can express this as $\mathbf{x} \in \mathbb{R}^n$.
The length of a vector is commonly called the *dimension* of the vector.

As with an ordinary Python array, we can access the length of an `ndarray`
by calling Python's built-in `len()` function.

```{.python .input  n=4}
len(x)
```

```{.json .output n=4}
[
 {
  "data": {
   "text/plain": "4"
  },
  "execution_count": 4,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

When an `ndarray` represents a vector (with precisely one axis),
we can also access its length via the `.shape` attribute.
The shape is a tuple that lists the length (dimensionality)
along each axis of the `ndarray`.
For `ndarray`s with just one axis, the shape has just one element.

```{.python .input  n=5}
x.shape
```

```{.json .output n=5}
[
 {
  "data": {
   "text/plain": "(4,)"
  },
  "execution_count": 5,
  "metadata": {},
  "output_type": "execute_result"
 }
]
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

In math notation, we use $\mathbf{A} \in \mathbb{R}^{m \times n}$
to express that the matrix $\mathbf{A}$ consists of $m$ rows and $n$ columns of real-valued scalars.
Visually, we can illustrate any matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ as a table,
where each element $a_{ij}$ belongs to the $i^{\mathrm{th}}$ row and $j^{\mathrm{th}}$ column:

$$\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \\ \end{bmatrix}.$$
:eqlabel:`eq_matrix_def`


For any $\mathbf{A} \in \mathbb{R}^{m \times n}$, the shape of $\mathbf{A}$
is ($m$, $n$) or $m \times n$.
We can create an $m \times n$ matrix in MXNet
by specifying a shape with two components $m$ and $n$
when calling any of our favorite functions for instantiating an `ndarray`.

```{.python .input  n=6}
A = np.arange(20).reshape(5, 4)
A
```

```{.json .output n=6}
[
 {
  "data": {
   "text/plain": "array([[ 0.,  1.,  2.,  3.],\n       [ 4.,  5.,  6.,  7.],\n       [ 8.,  9., 10., 11.],\n       [12., 13., 14., 15.],\n       [16., 17., 18., 19.]])"
  },
  "execution_count": 6,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

We can access the scalar element $a_{ij}$ of a matrix $\mathbf{A}$ in :eqref:`eq_matrix_def`
by specifying the indices for the row ($i$) and column ($j$),
such as $[\mathbf{A}]_{ij}$.
When the scalar elements of a matrix $\mathbf{A}$, such as in :eqref:`eq_matrix_def`, are not given,
we may simply use the lower-case letter of the matrix $\mathbf{A}$ with the index subscript, $a_{ij}$,
to refer to $[\mathbf{A}]_{ij}$.
To keep notation simple, commas are inserted to separate indices only when necessary,
such as $a_{2,3j}$ and $[\mathbf{A}]_{2i-1,3}$.


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

In code, we access a matrix's transpose via the `T` attribute.

```{.python .input  n=7}
A.T
```

```{.json .output n=7}
[
 {
  "data": {
   "text/plain": "array([[ 0.,  4.,  8., 12., 16.],\n       [ 1.,  5.,  9., 13., 17.],\n       [ 2.,  6., 10., 14., 18.],\n       [ 3.,  7., 11., 15., 19.]])"
  },
  "execution_count": 7,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Matrices are useful data structures: 
they allow us to organize data that have different modalities of variation. 
For example, rows in our matrix might correspond to different houses (data points), 
while columns might correspond to different attributes.
This should sound familiar if you have ever used spreadsheet software or
have read :ref:`sec_pandas`.
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
Tensors are denoted with capital letters of a special font face
(e.g., $\mathsf{X}$, $\mathsf{Y}$, and $\mathsf{Z}$)
and their indexing mechanism (e.g., $x_{ijk}$ and $[\mathsf{X}]_{1, 2i-1,3}$) is similar to that of matrices. 

Tensors will become more important when we start working with images, which arrive as `ndarray`s with 3 axes corresponding to the height, width, and a *channel* axis for stacking the color channels (red, green, and blue). For now, we will skip over higher order tensors and focus on the basics.

```{.python .input  n=9}
X = np.arange(24).reshape(2, 3, 4)
X
```

```{.json .output n=9}
[
 {
  "data": {
   "text/plain": "(array([[[ 0.,  1.,  2.,  3.],\n         [ 4.,  5.,  6.,  7.],\n         [ 8.,  9., 10., 11.]],\n \n        [[12., 13., 14., 15.],\n         [16., 17., 18., 19.],\n         [20., 21., 22., 23.]]]), 2)"
  },
  "execution_count": 9,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## Summary

* Scalars, vectors, matrices, and tensors are basic mathematical objects in linear algebra.
* Vectors generalize scalars, and matrices generalize vectors.
* In the `ndarray` representation, scalars, vectors, matrices, and tensors have 0, 1, 2, and an arbitrary number of axes, respectively.


## Exercises

1. We defined the tensor `X` of shape ($2$, $3$, $4$) in this section. What is the output of `len(X)`?
2. For a tensor `X` of arbitrary shape, does `len(X)` always correspond to the length of a certain axis of `X`? What is that axis?


## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2317)

![](../img/qr_scalar-tensor.svg)
