{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": true
   },
   "source": [
    "# Linear algebra\n",
    "\n",
    "Now that you can store and manipulate data, \n",
    "let's briefly review the subset of basic linear algebra \n",
    "that you'll need to understand most of the models. \n",
    "We'll introduce all the basic concepts, \n",
    "the corresponding mathematical notation, \n",
    "and their realization in code all in one place. \n",
    "If you're already confident in your basic linear algebra, \n",
    "feel free to skim or skip this chapter. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from mxnet import nd"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Scalars\n",
    "\n",
    "If you never studied linear algebra or machine learning, \n",
    "you're probably used to working with one number at a time.\n",
    "And know how to do basic things like add them together or multiply them.\n",
    "For example, in Palo Alto, the temperature is $52$ degrees Fahrenheit. \n",
    "Formally, we call these values $scalars$.\n",
    "If you wanted to convert this value to Celsius (using metric system's more sensible unit of temperature measurement),\n",
    "you'd evaluate the expression $c = (f - 32) * 5/9$ setting $f$ to $52$.\n",
    "In this equation, each of the terms $32$, $5$, and $9$ is a scalar value.\n",
    "The placeholders $c$ and $f$ that we use are called variables\n",
    "and they stand in for unknown scalar values.\n",
    "\n",
    "In mathematical notation, we represent scalars with ordinary lower cased letters ($x$, $y$, $z$).\n",
    "We also denote the space of all scalars as $\\mathcal{R}$.\n",
    "For expedience, we're going to punt a bit on what precisely a space is,\n",
    "but for now, remember that if you want to say that $x$ is a scalar, \n",
    "you can simply say $x \\in \\mathcal{R}$.\n",
    "The symbol $\\in$ can be pronounced \"in\" and just denotes membership in a set.\n",
    "\n",
    "In MXNet, we work with scalars by creating NDArrays with just one element. \n",
    "In this snippet, we instantiate two scalars and perform some familiar arithmetic operations with them."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "x + y =  \n",
      "[ 5.]\n",
      "<NDArray 1 @cpu(0)>\n",
      "x * y =  \n",
      "[ 6.]\n",
      "<NDArray 1 @cpu(0)>\n",
      "x / y =  \n",
      "[ 1.5]\n",
      "<NDArray 1 @cpu(0)>\n",
      "x ** y =  \n",
      "[ 9.]\n",
      "<NDArray 1 @cpu(0)>\n"
     ]
    }
   ],
   "source": [
    "##########################\n",
    "# Instantiate two scalars\n",
    "##########################\n",
    "x = nd.array([3.0]) \n",
    "y = nd.array([2.0])\n",
    "\n",
    "##########################\n",
    "# Add them\n",
    "##########################\n",
    "print('x + y = ', x + y)\n",
    "\n",
    "##########################\n",
    "# Multiply them\n",
    "##########################\n",
    "print('x * y = ', x * y)\n",
    "\n",
    "##########################\n",
    "# Divide x by y\n",
    "##########################\n",
    "print('x / y = ', x / y)\n",
    "\n",
    "##########################\n",
    "# Raise x to the power y. \n",
    "##########################\n",
    "print('x ** y = ', nd.power(x,y))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can convert any NDArray to a Python float by calling its `asscalar` method"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "3.0"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x.asscalar()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Vectors \n",
    "You can think of a vector as simply a list of numbers, for example ``[1.0,3.0,4.0,2.0]``. \n",
    "Each of the numbers in the vector consists of a single scalar value.\n",
    "We call these values the *entries* or *components* of the vector.\n",
    "Often, we're interested in vectors whose values hold some real-world significance.\n",
    "For example, if we're studying the risk that loans default,\n",
    "we might associate each applicant with a vector \n",
    "whose components correspond to their income, \n",
    "length of employment, number of previous defaults, etc. \n",
    "If we were studying the risk of heart attack in hospital patients, \n",
    "we might represent each patient with a vector\n",
    "whose components capture their most recent vital signs,\n",
    "cholesterol levels, minutes of exercise per day, etc. \n",
    "In math notation, we'll usually denote vectors as bold-faced, \n",
    "lower-cased letters ($\\mathbf{u}$, $\\mathbf{v}$, $\\mathbf{w})$. \n",
    "In MXNet, we work with vectors via 1D NDArrays with an arbitrary number of components."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "u =  \n",
      "[ 0.  1.  2.  3.]\n",
      "<NDArray 4 @cpu(0)>\n"
     ]
    }
   ],
   "source": [
    "u = nd.arange(4)\n",
    "print('u = ', u)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can refer to any element of a vector by using a subscript. \n",
    "For example, we can refer to the $4$th element of $\\mathbf{u}$ by $u_4$. \n",
    "Note that the element $u_4$ is a scalar, \n",
    "so we don't bold-face the font when referring to it.\n",
    "In code, we access any element $i$ by indexing into the ``NDArray``."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\n",
       "[ 3.]\n",
       "<NDArray 1 @cpu(0)>"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "u[3]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Length, dimensionality, and, shape\n",
    "\n",
    "A vector is just an array of numbers. And just as every array has a length, so does every vector. \n",
    "In math notation, if we want to say that a vector $x$ consists of $n$ real-valued scalars,\n",
    "we can express this as $\\mathbf{x} \\in \\mathcal{R}^n$.\n",
    "The length of a vector is commonly called its $dimension$.\n",
    "As with an ordinary Python array, we can access the length of an NDArray \n",
    "by calling Python's in-built ``len()`` function."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "4"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(u)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can also access a vector's length via its `.shape` attribute. \n",
    "The shape is a tuple that lists the dimensionality of the NDArray along each of its axes. \n",
    "Because a vector can only be indexed along one axis, its shape has just one element."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(4,)"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "u.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Note that the word dimension is overloaded and this tends to confuse people.\n",
    "Some use the *dimensionality* of a vector to refer to its length (the number of components). \n",
    "However some use the word *dimensionality* to refer to the number of axes that an array has.\n",
    "In this sense, a scalar *would have* $0$ dimensions and a vector *would have* $1$ dimension.\n",
    "**To avoid confusion, when we say *2D* array or *3D* array, we mean an array with 2 or 3 axes repespectively. But if we say *$n$-dimensional* vector, we mean a vector of length $n$.**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "a = 2\n",
    "x = nd.array([1,2,3])\n",
    "y = nd.array([10,20,30])\n",
    "print(a * x)\n",
    "print(a * x + y)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Matrices\n",
    "\n",
    "Just as vectors generalize scalars from order $0$ to order $1$, \n",
    "matrices generalize vectors from $1D$ to $2D$. \n",
    "Matrices, which we'll denote with capital letters ($A$, $B$, $C$), \n",
    "are represented in code as arrays with 2 axes.  \n",
    "Visually, we can draw a matrix as a table,\n",
    "where each entry $a_{ij}$ belongs to the $i$-th row and $j$-th column. \n",
    "\n",
    "\n",
    "$$A=\\begin{pmatrix}\n",
    " a_{11} & a_{12} & \\cdots & a_{1m} \\\\\n",
    " a_{21} & a_{22} & \\cdots & a_{2m} \\\\\n",
    "\\vdots & \\vdots & \\ddots & \\vdots \\\\\n",
    " a_{n1} & a_{n2} & \\cdots & a_{nm} \\\\\n",
    "\\end{pmatrix}$$\n",
    "\n",
    "We can create a matrix with $n$ rows and $m$ columns in MXNet\n",
    "by specifying a shape with two components `(n,m)`\n",
    "when calling any of our favorite functions for instantiating an `ndarray`\n",
    "such as `ones`, or `zeros`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\n",
       "[[ 0.  0.  0.  0.]\n",
       " [ 0.  0.  0.  0.]\n",
       " [ 0.  0.  0.  0.]\n",
       " [ 0.  0.  0.  0.]\n",
       " [ 0.  0.  0.  0.]]\n",
       "<NDArray 5x4 @cpu(0)>"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "A = nd.zeros((5,4))\n",
    "A"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can also reshape any 1D array into a 2D ndarray by calling `ndarray`'s reshape method and passing in the desired shape. Note that the product of shape components `n * m` must be equal to the length of the original vector. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\n",
       "[[  0.   1.   2.   3.]\n",
       " [  4.   5.   6.   7.]\n",
       " [  8.   9.  10.  11.]\n",
       " [ 12.  13.  14.  15.]\n",
       " [ 16.  17.  18.  19.]]\n",
       "<NDArray 5x4 @cpu(0)>"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x = nd.arange(20)\n",
    "A = x.reshape((5, 4))\n",
    "A"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Matrices are useful data structures: they allow us to organize data that has different modalities of variation. For example, returning to the example of medical data, rows in our matrix might correspond to different patients, while columns might correspond to different attributes."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can access the scalar elements $a_{ij}$ of a matrix $A$ by specifying the indices for the row ($i$) and column ($j$) respectively. Let's grab the element $a_{2,3}$ from the random matrix we initialized above."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "A[2, 3] =  \n",
      "[ 11.]\n",
      "<NDArray 1 @cpu(0)>\n"
     ]
    }
   ],
   "source": [
    "print('A[2, 3] = ', A[2, 3])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can also grab the vectors corresponding to an entire row $\\mathbf{a}_{i,:}$ or a column $\\mathbf{a}_{:,j}$."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "row 2 \n",
      "[  8.   9.  10.  11.]\n",
      "<NDArray 4 @cpu(0)>\n",
      "column 3 \n",
      "[  3.   7.  11.  15.  19.]\n",
      "<NDArray 5 @cpu(0)>\n"
     ]
    }
   ],
   "source": [
    "print('row 2', A[2, :])\n",
    "print('column 3', A[:, 3])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can transpose the matrix through `T`. That is, if $B = A^T$, then $b_{ij} = a_{ji}$ for any $i$ and $j$."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\n",
       "[[  0.   4.   8.  12.  16.]\n",
       " [  1.   5.   9.  13.  17.]\n",
       " [  2.   6.  10.  14.  18.]\n",
       " [  3.   7.  11.  15.  19.]]\n",
       "<NDArray 4x5 @cpu(0)>"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "A.T"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Tensors \n",
    "\n",
    "Just as vectors generalize scalars, and matrices generalize vectors, we can actually build data structures with even more axes. Tensors give us a generic way of discussing arrays with an arbitrary number of axes. Vectors, for example, are first-order tensors, and matrices are second-order tensors.\n",
    "\n",
    "Using tensors will become more important when we start working with images, which arrive as 3D data structures, with axes corresponding to the height, width, and the three (RGB) color channels. But in this chapter, we're going to skip past and make sure you know the basics."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "X.shape = (2, 3, 4)\n",
      "X = \n",
      "[[[  0.   1.   2.   3.]\n",
      "  [  4.   5.   6.   7.]\n",
      "  [  8.   9.  10.  11.]]\n",
      "\n",
      " [[ 12.  13.  14.  15.]\n",
      "  [ 16.  17.  18.  19.]\n",
      "  [ 20.  21.  22.  23.]]]\n",
      "<NDArray 2x3x4 @cpu(0)>\n"
     ]
    }
   ],
   "source": [
    "X = nd.arange(24).reshape((2, 3, 4))\n",
    "print('X.shape =', X.shape)\n",
    "print('X =', X)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Element-wise operations\n",
    "\n",
    "Oftentimes, we want to apply functions to arrays. \n",
    "Some of the simplest and most useful functions are the element-wise functions. \n",
    "These operate by performing a single scalar operation on the corresponding elements of two arrays.\n",
    "We can create an element-wise function from any function that maps from the scalars to the scalars.\n",
    "In math notations we would denote such a function as $f: \\mathcal{R} \\rightarrow \\mathcal{R}$.\n",
    "Given any two vectors $\\mathbf{u}$ and $\\mathbf{v}$ *of the same shape*, and the function f,\n",
    "we can produce a vector $\\mathbf{c} = F(\\mathbf{u},\\mathbf{v})$ \n",
    "by setting $c_i \\gets f(u_i, v_i)$ for all $i$.\n",
    "Here, we produced the vector-valued $F: \\mathcal{R}^d \\rightarrow \\mathcal{R}^d$\n",
    "by *lifting* the scalar function to an element-wise vector operation.\n",
    "In MXNet, the common standard arithmetic operators (+,-,/,\\*,\\*\\*)\n",
    "have all been *lifted* to element-wise operations for identically-shaped tensors of arbitrary shape."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "v = \n",
      "[ 2.  2.  2.  2.]\n",
      "<NDArray 4 @cpu(0)>\n",
      "u + v \n",
      "[  3.   4.   6.  10.]\n",
      "<NDArray 4 @cpu(0)>\n",
      "u - v \n",
      "[-1.  0.  2.  6.]\n",
      "<NDArray 4 @cpu(0)>\n",
      "u * v \n",
      "[  2.   4.   8.  16.]\n",
      "<NDArray 4 @cpu(0)>\n",
      "u / v \n",
      "[ 0.5  1.   2.   4. ]\n",
      "<NDArray 4 @cpu(0)>\n"
     ]
    }
   ],
   "source": [
    "u = nd.array([1, 2, 4, 8])\n",
    "v = nd.ones_like(u) * 2\n",
    "print('v =', v)\n",
    "print('u + v', u + v)\n",
    "print('u - v', u - v)\n",
    "print('u * v', u * v)\n",
    "print('u / v', u / v)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can call element-wise operations on any two tensors of the same shape, including matrices."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "B = \n",
      "[[ 3.  3.  3.  3.]\n",
      " [ 3.  3.  3.  3.]\n",
      " [ 3.  3.  3.  3.]\n",
      " [ 3.  3.  3.  3.]\n",
      " [ 3.  3.  3.  3.]]\n",
      "<NDArray 5x4 @cpu(0)>\n",
      "A + B = \n",
      "[[  3.   4.   5.   6.]\n",
      " [  7.   8.   9.  10.]\n",
      " [ 11.  12.  13.  14.]\n",
      " [ 15.  16.  17.  18.]\n",
      " [ 19.  20.  21.  22.]]\n",
      "<NDArray 5x4 @cpu(0)>\n",
      "A * B = \n",
      "[[  0.   3.   6.   9.]\n",
      " [ 12.  15.  18.  21.]\n",
      " [ 24.  27.  30.  33.]\n",
      " [ 36.  39.  42.  45.]\n",
      " [ 48.  51.  54.  57.]]\n",
      "<NDArray 5x4 @cpu(0)>\n"
     ]
    }
   ],
   "source": [
    "B = nd.ones_like(A) * 3\n",
    "print('B =', B)\n",
    "print('A + B =', A + B)\n",
    "print('A * B =', A * B)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Basic properties of tensor arithmetic\n",
    "\n",
    "Scalars, vectors, matrices, and tensors of any order have some nice properties that we'll often rely on.\n",
    "For example, as you might have noticed from the definition of an element-wise operation, \n",
    "given operands with the same shape, \n",
    "the result of any element-wise operation is a tensor of that same shape. \n",
    "Another convenient property is that for all tensors, multiplication by a scalar \n",
    "produces a tensor of the same shape. \n",
    "In math, given two tensors $X$ and $Y$ with the same shape,\n",
    "$\\alpha X + Y$ has the same shape. \n",
    "(numerical mathematicians call this the AXPY operation). "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(3,)\n",
      "(3,)\n",
      "(3,)\n",
      "(3,)\n"
     ]
    }
   ],
   "source": [
    "a = 2\n",
    "x = nd.ones(3)\n",
    "y = nd.zeros(3)\n",
    "print(x.shape)\n",
    "print(y.shape)\n",
    "print((a * x).shape)\n",
    "print((a * x + y).shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Shape is not the the only property preserved under addition and multiplication by a scalar. These operations also preserve membership in a vector space. But we'll postpone this discussion for the second half of this chapter because it's not critical to getting your first models up and running. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Sums and means \n",
    "\n",
    "The next more sophisticated thing we can do with arbitrary tensors \n",
    "is to calculate the sum of their elements. \n",
    "In mathematical notation, we express sums using the $\\sum$ symbol. \n",
    "To express the sum of the elements in a vector $\\mathbf{u}$ of length $d$, \n",
    "we can write $\\sum_{i=1}^d u_i$. In code, we can just call ``nd.sum()``."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "nd.sum(u)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can similarly express sums over the elements of tensors of arbitrary shape. For example, the sum of the elements of an $m \\times n$ matrix $A$ could be written $\\sum_{i=1}^{m} \\sum_{j=1}^{n} a_{ij}$. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "nd.sum(A)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "A related quantity is the *mean*, which is also called the *average*. \n",
    "We calculate the mean by dividing the sum by the total number of elements. \n",
    "With mathematical notation, we could write the average \n",
    "over a vector $\\mathbf{u}$ as $\\frac{1}{d} \\sum_{i=1}^{d} u_i$ \n",
    "and the average over a matrix $A$ as  $\\frac{1}{n \\cdot m} \\sum_{i=1}^{m} \\sum_{j=1}^{n} a_{ij}$. \n",
    "In code, we could just call ``nd.mean()`` on tensors of arbitrary shape:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "print(nd.mean(A))\n",
    "print(nd.sum(A) / A.size)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Dot products\n",
    "\n",
    "<!-- So far, we've only performed element-wise operations, sums and averages. And if this was we could do, linear algebra probably wouldn't deserve it's own chapter. However, -->\n",
    "\n",
    "One of the most fundamental operations is the dot product. Given two vectors $\\mathbf{u}$ and $\\mathbf{v}$, the dot product $\\mathbf{u}^T \\mathbf{v}$ is a sum over the products of the corresponding elements: $\\mathbf{u}^T \\mathbf{v} = \\sum_{i=1}^{d} u_i \\cdot v_i$."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "nd.dot(u, v)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Note that we can express the dot product of two vectors ``nd.dot(u, v)`` equivalently by performing an element-wise multiplication and then a sum:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "nd.sum(u * v)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Dot products are useful in a wide range of contexts. For example, given a set of weights $\\mathbf{w}$, the weighted sum of some values ${u}$ could be expressed as the dot product $\\mathbf{u}^T \\mathbf{w}$. When the weights are non-negative and sum to one ($\\sum_{i=1}^{d} {w_i} = 1$), the dot product expresses a *weighted average*. When two vectors each have length one (we'll discuss what *length* means below in the section on norms), dot products can also capture the cosine of the angle between them."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Matrix-vector products\n",
    "\n",
    "Now that we know how to calculate dot products we can begin to understand matrix-vector products. Let's start off by visualizing a matrix $A$ and a column vector $\\mathbf{x}$.\n",
    "\n",
    "$$A=\\begin{pmatrix}\n",
    " a_{11} & a_{12} & \\cdots & a_{1m} \\\\\n",
    " a_{21} & a_{22} & \\cdots & a_{2m} \\\\\n",
    "\\vdots & \\vdots & \\ddots & \\vdots \\\\\n",
    " a_{n1} & a_{n2} & \\cdots & a_{nm} \\\\\n",
    "\\end{pmatrix},\\quad\\mathbf{x}=\\begin{pmatrix}\n",
    " x_{1}  \\\\\n",
    " x_{2} \\\\\n",
    "\\vdots\\\\\n",
    " x_{m}\\\\\n",
    "\\end{pmatrix} $$\n",
    "\n",
    "We can visualize the matrix in terms of its row vectors\n",
    "\n",
    "$$A=\n",
    "\\begin{pmatrix}\n",
    "\\cdots & \\mathbf{a}^T_{1} &...  \\\\\n",
    "\\cdots & \\mathbf{a}^T_{2} & \\cdots \\\\\n",
    " & \\vdots &  \\\\\n",
    " \\cdots &\\mathbf{a}^T_n & \\cdots \\\\\n",
    "\\end{pmatrix},$$\n",
    "\n",
    "where each $\\mathbf{a}^T_{i} \\in \\mathbb{R}^{m}$\n",
    "is a row vector representing the $i$-th row of the matrix $A$.\n",
    "\n",
    "Then the matrix vector product $\\mathbf{y} = A\\mathbf{x}$ is simply a column vector $\\mathbf{y} \\in \\mathbb{R}^n$ where each entry $y_i$ is the dot product $\\mathbf{a}^T_i \\mathbf{x}$.\n",
    "\n",
    "$$A\\mathbf{x}=\n",
    "\\begin{pmatrix}\n",
    "\\cdots & \\mathbf{a}^T_{1} &...  \\\\\n",
    "\\cdots & \\mathbf{a}^T_{2} & \\cdots \\\\\n",
    " & \\vdots &  \\\\\n",
    " \\cdots &\\mathbf{a}^T_n & \\cdots \\\\\n",
    "\\end{pmatrix}\n",
    "\\begin{pmatrix}\n",
    " x_{1}  \\\\\n",
    " x_{2} \\\\\n",
    "\\vdots\\\\\n",
    " x_{m}\\\\\n",
    "\\end{pmatrix}\n",
    "= \\begin{pmatrix}\n",
    " \\mathbf{a}^T_{1} \\mathbf{x}  \\\\\n",
    " \\mathbf{a}^T_{2} \\mathbf{x} \\\\\n",
    "\\vdots\\\\\n",
    " \\mathbf{a}^T_{n} \\mathbf{x}\\\\\n",
    "\\end{pmatrix}\n",
    "$$\n",
    "\n",
    "So you can think of multiplication by a matrix $A\\in \\mathbb{R}^{m \\times n}$ as a transformation that projects vectors from $\\mathbb{R}^{m}$ to $\\mathbb{R}^{n}$.\n",
    "\n",
    "These transformations turn out to be quite useful. For example, we can represent rotations as multiplications by a square matrix. As we'll see in subsequent chapters, we can also use matrix-vector products to describe the calculations of each layer in a neural network. \n",
    "\n",
    "Expressing matrix-vector products in code with ``ndarray``, we use the same ``nd.dot()`` function as for dot products. When we call ``nd.dot(A, x)`` with a matrix ``A`` and a vector ``x``, ``MXNet`` knows to perform a matrix-vector product. Note that the column dimension of ``A`` must be the same as the dimension of ``x``."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "nd.dot(A, u)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Matrix-matrix multiplication\n",
    "\n",
    "If you've gotten the hang of dot products and matrix-vector multiplication, then matrix-matrix multiplications should be pretty straightforward.\n",
    "\n",
    "Say we have two matrices, $A \\in \\mathbb{R}^{n \\times k}$ and $B \\in \\mathbb{R}^{k \\times m}$:\n",
    "\n",
    "$$A=\\begin{pmatrix}\n",
    " a_{11} & a_{12} & \\cdots & a_{1k} \\\\\n",
    " a_{21} & a_{22} & \\cdots & a_{2k} \\\\\n",
    "\\vdots & \\vdots & \\ddots & \\vdots \\\\\n",
    " a_{n1} & a_{n2} & \\cdots & a_{nk} \\\\\n",
    "\\end{pmatrix},\\quad\n",
    "B=\\begin{pmatrix}\n",
    " b_{11} & b_{12} & \\cdots & b_{1m} \\\\\n",
    " b_{21} & b_{22} & \\cdots & b_{2m} \\\\\n",
    "\\vdots & \\vdots & \\ddots & \\vdots \\\\\n",
    " b_{k1} & b_{k2} & \\cdots & b_{km} \\\\\n",
    "\\end{pmatrix}$$\n",
    "\n",
    "To produce the matrix product $C = AB$, it's easiest to think of $A$ in terms of its row vectors and $B$ in terms of its column vectors:\n",
    "\n",
    "$$A=\n",
    "\\begin{pmatrix}\n",
    "\\cdots & \\mathbf{a}^T_{1} &...  \\\\\n",
    "\\cdots & \\mathbf{a}^T_{2} & \\cdots \\\\\n",
    " & \\vdots &  \\\\\n",
    " \\cdots &\\mathbf{a}^T_n & \\cdots \\\\\n",
    "\\end{pmatrix},\n",
    "\\quad B=\\begin{pmatrix}\n",
    "\\vdots & \\vdots &  & \\vdots \\\\\n",
    " \\mathbf{b}_{1} & \\mathbf{b}_{2} & \\cdots & \\mathbf{b}_{m} \\\\\n",
    " \\vdots & \\vdots &  &\\vdots\\\\\n",
    "\\end{pmatrix}.\n",
    "$$\n",
    "\n",
    "Note here that each row vector $\\mathbf{a}^T_{i}$ lies in $\\mathbb{R}^k$ and that each column vector $\\mathbf{b}_j$ also lies in $\\mathbb{R}^k$.\n",
    "\n",
    "Then to produce the matrix product $C \\in \\mathbb{R}^{n \\times m}$ we simply compute each entry $c_{ij}$ as the dot product $\\mathbf{a}^T_i \\mathbf{b}_j$.\n",
    "\n",
    "$$C = AB = \\begin{pmatrix}\n",
    "\\cdots & \\mathbf{a}^T_{1} &...  \\\\\n",
    "\\cdots & \\mathbf{a}^T_{2} & \\cdots \\\\\n",
    " & \\vdots &  \\\\\n",
    " \\cdots &\\mathbf{a}^T_n & \\cdots \\\\\n",
    "\\end{pmatrix}\n",
    "\\begin{pmatrix}\n",
    "\\vdots & \\vdots &  & \\vdots \\\\\n",
    " \\mathbf{b}_{1} & \\mathbf{b}_{2} & \\cdots & \\mathbf{b}_{m} \\\\\n",
    " \\vdots & \\vdots &  &\\vdots\\\\\n",
    "\\end{pmatrix}\n",
    "= \\begin{pmatrix}\n",
    "\\mathbf{a}^T_{1} \\mathbf{b}_1 & \\mathbf{a}^T_{1}\\mathbf{b}_2& \\cdots & \\mathbf{a}^T_{1} \\mathbf{b}_m \\\\\n",
    " \\mathbf{a}^T_{2}\\mathbf{b}_1 & \\mathbf{a}^T_{2} \\mathbf{b}_2 & \\cdots & \\mathbf{a}^T_{2} \\mathbf{b}_m \\\\\n",
    " \\vdots & \\vdots & \\ddots &\\vdots\\\\\n",
    "\\mathbf{a}^T_{n} \\mathbf{b}_1 & \\mathbf{a}^T_{n}\\mathbf{b}_2& \\cdots& \\mathbf{a}^T_{n} \\mathbf{b}_m \n",
    "\\end{pmatrix}\n",
    "$$\n",
    "\n",
    "You can think of the matrix-matrix multiplication $AB$ as simply performing $m$ matrix-vector products and stitching the results together to form an $n \\times m$ matrix. Just as with ordinary dot products and matrix-vector products, we can compute matrix-matrix products in ``MXNet`` by using ``nd.dot()``."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\n",
       "[[ 4.  4.  4.  4.  4.]\n",
       " [ 4.  4.  4.  4.  4.]\n",
       " [ 4.  4.  4.  4.  4.]]\n",
       "<NDArray 3x5 @cpu(0)>"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "A = nd.ones(shape=(3, 4))\n",
    "B = nd.ones(shape=(4, 5))\n",
    "nd.dot(A, B)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": true
   },
   "source": [
    "## Norms"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": true
   },
   "source": [
    "Before we can start implementing models, \n",
    "there's one last concept we're going to introduce. \n",
    "Some of the most useful operators in linear algebra are norms.\n",
    "Informally, they tell us how big a vector or matrix is. \n",
    "We represent norms with the notation $\\|\\cdot\\|$. \n",
    "The $\\cdot$ in this expression is just a placeholder. \n",
    "For example, we would represent the norm of a vector $\\mathbf{x}$ \n",
    "or matrix $A$ as $\\|\\mathbf{x}\\|$ or $\\|A\\|$, respectively. \n",
    "\n",
    "All norms must satisfy a handful of properties:\n",
    "1. $\\|\\alpha A\\| = |\\alpha| \\|A\\|$\n",
    "2. $\\|A + B\\| \\leq \\|A\\| + \\|B\\|$\n",
    "3. $\\|A\\| \\geq 0$\n",
    "4. If $\\forall {i,j}, a_{ij} = 0$, then $\\|A\\|=0$\n",
    "\n",
    "To put it in words, the first rule says \n",
    "that if we scale all the components of a matrix or vector \n",
    "by a constant factor $\\alpha$, \n",
    "its norm also scales by the *absolute value* \n",
    "of the same constant factor. \n",
    "The second rule is the familiar triangle inequality.\n",
    "The third rule simple says that the norm must be non-negative. \n",
    "That makes sense, in most contexts the smallest *size* for anything is 0.\n",
    "The final rule basically says that the smallest norm is achieved by a matrix or vector consisting of all zeros.\n",
    "It's possible to define a norm that gives zero norm to nonzero matrices,\n",
    "but you can't give nonzero norm to zero matrices. \n",
    "That's a mouthful, but if you digest it then you probably have grepped the important concepts here.\n",
    "\n",
    "If you remember Euclidean distances (think Pythagoras' theorem) from grade school, \n",
    "then non-negativity and the triangle inequality might ring a bell.\n",
    "You might notice that norms sound a lot like measures of distance.\n",
    "\n",
    "In fact, the Euclidean distance $\\sqrt{x_1^2 + \\cdots + x_n^2}$ is a norm. \n",
    "Specifically it's the $\\ell_2$-norm. \n",
    "An analogous computation, \n",
    "performed over the entries of a matrix, e.g. $\\sqrt{\\sum_{i,j} a_{ij}^2}$, \n",
    "is called the Frobenius norm. \n",
    "More often, in machine learning we work with the squared $\\ell_2$ norm (notated $\\ell_2^2$).\n",
    "We also commonly work with the $\\ell_1$ norm.\n",
    "The $\\ell_1$ norm is simply the sum of the absolute values. \n",
    "It has the convenient property of placing less emphasis on outliers.\n",
    "\n",
    "To calculate the $\\ell_2$ norm, we can just call ``nd.norm()``.  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "nd.norm(u)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To calculate the L1-norm we can simply perform the absolute value and then sum over the elements."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "nd.sum(nd.abs(u))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Norms and objectives\n",
    "\n",
    "While we don't want to get too far ahead of ourselves, we do want you to anticipate why these concepts are useful.\n",
    "In machine learning we're often trying to solve optimization problems: *Maximize* the probability assigned to observed data. *Minimize* the distance between predictions and the ground-truth observations. Assign vector representations to items (like words, products, or news articles) such that the distance between similar items is minimized, and the distance between dissimilar items is maximized. Oftentimes, these objectives, perhaps the most important component of a machine learning algorithm (besides the data itself), are expressed as norms.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Intermediate linear algebra\n",
    "\n",
    "If you've made it this far, and understand everything that we've covered,\n",
    "then honestly, you *are* ready to begin modeling. \n",
    "If you're feeling antsy, this is a perfectly reasonable place to move on.\n",
    "You already know nearly all of the linear algebra required \n",
    "to implement a number of many practically useful models\n",
    "and you can always circle back when you want to learn more.\n",
    "\n",
    "But there's a lot more to linear algebra, even as concerns machine learning. \n",
    "At some point, if you plan to make a career of machine learning,\n",
    "you'll need to know more than we've covered so far. \n",
    "In the rest of this chapter, we introduce some useful, more advanced concepts.\n",
    "\n",
    "\n",
    "\n",
    "## Basic vector properties\n",
    "\n",
    "Vectors are useful beyond being data structures to carry numbers.\n",
    "In addition to reading and writing values to the components of a vector,\n",
    "and performing some useful mathematical operations,\n",
    "we can analyze vectors in some interesting ways.\n",
    "\n",
    "One important concept is the notion of a vector space.\n",
    "Here are the conditions that make a vector space:\n",
    "\n",
    "* **Additive axioms** (we assume that x,y,z are all vectors): \n",
    "  $x+y = y+x$ and $(x+y)+z = x+(y+z)$ and $0+x = x+0 = x$ and $(-x) + x = x + (-x) = 0$.\n",
    "* **Multiplicative axioms** (we assume that x is a vector and a, b are scalars):\n",
    "  $0 \\cdot x = 0$ and $1 \\cdot x = x$ and $(a b) x = a (b x)$.\n",
    "* **Distributive axioms** (we assume that x and y are vectors and a, b are scalars):\n",
    "  $a(x+y) = ax + ay$ and $(a+b)x = ax +bx$."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Special matrices \n",
    "\n",
    "There are a number of special matrices that we will use throughout this tutorial. Let's look at them in a bit of detail:\n",
    "\n",
    "* **Symmetric Matrix** These are matrices where the entries below and above the diagonal are the same. In other words, we have that $M^\\top = M$. An example of such matrices are those that describe pairwise distances, i.e. $M_{ij} = \\|x_i - x_j\\|$. Likewise, the Facebook friendship graph can be written as a symmetric matrix where $M_{ij} = 1$ if $i$ and $j$ are friends and $M_{ij} = 0$ if they are not. Note that the *Twitter* graph is asymmetric - $M_{ij} = 1$, i.e. $i$ following $j$ does not imply that $M_{ji} = 1$, i.e. $j$ following $i$.\n",
    "* **Antisymmetric Matrix** These matrices satisfy $M^\\top = -M$. Note that any arbitrary matrix can always be decomposed into a symmetric and into an antisymmetric matrix by using $M = \\frac{1}{2}(M + M^\\top) + \\frac{1}{2}(M - M^\\top)$. \n",
    "* **Diagonally Dominant Matrix** These are matrices where the off-diagonal elements are small relative to the main diagonal elements. In particular we have that $M_{ii} \\geq \\sum_{j \\neq i} M_{ij}$ and $M_{ii} \\geq \\sum_{j \\neq i} M_{ji}$. If a matrix has this property, we can often approximate $M$ by its diagonal. This is often expressed as $\\mathrm{diag}(M)$. \n",
    "* **Positive Definite Matrix** These are matrices that have the nice property where $x^\\top M x > 0$ whenever $x \\neq 0$. Intuitively, they are a generalization of the squared norm of a vector $\\|x\\|^2 = x^\\top x$. It is easy to check that whenever $M = A^\\top A$, this holds since there $x^\\top M x = x^\\top A^\\top A x = \\|A x\\|^2$. There is a somewhat more profound theorem which states that all positive definite matrices can be written in this form. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": true
   },
   "source": [
    "## Conclusions\n",
    "\n",
    "In just a few pages (or one Jupyter notebook) we've taught you all the linear algebra you'll need to understand a good chunk of neural networks. Of course there's a *lot* more to linear algebra. And a lot of that math *is* useful for machine learning. For example, matrices can be decomposed into factors, and these decompositions can reveal low-dimensional structure in real-world datasets. There are entire subfields of machine learning that focus on using matrix decompositions and their generalizations to high-order tensors to discover structure in datasets and solve prediction problems. But this book focuses on deep learning. And we believe you'll be much more inclined to learn more mathematics once you've gotten your hands dirty deploying useful machine learning models on real datasets. So while we reserve the right to introduce more math much later on, we'll wrap up this chapter here.\n",
    "\n",
    "If you're eager to learn more about linear algebra, here are some of our favorite resources on the topic\n",
    "* For a solid primer on basics, check out Gilbert Strang's book [Introduction to Linear Algebra](http://math.mit.edu/~gs/linearalgebra/)\n",
    "* Zico Kolter's [Linear Algebra Reivew and Reference](http://www.cs.cmu.edu/~zkolter/course/15-884/linalg-review.pdf)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Next\n",
    "[Probability and statistics](../chapter01_crashcourse/probability.ipynb)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "For whinges or inquiries, [open an issue on  GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.4.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
