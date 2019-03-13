# Mathematical Basis

This section summarizes the basic knowledge of linear algebra, differentiation, and probability required to understand the contents in this book. To avoid long discussions of mathematical knowledge not required to understand this book, a few definitions in this section are slightly simplified.


## Linear Algebra

Below we summarize the concepts of vectors, matrices, operations, norms, eigenvectors, and eigenvalues.

### Vectors

Vectors in this book refer to column vectors. An $n$-dimensional vector $\boldsymbol{x}$ can be written as

$$
\boldsymbol{x} =
\begin{bmatrix}
    x_{1}  \\
    x_{2}  \\
    \vdots  \\
    x_{n}
\end{bmatrix},
$$

where $x_1, \ldots, x_n$ are elements of the vector. To express that $\boldsymbol{x}$ is an $n$-dimensional vector with elements from the set of real numbers, we write $\boldsymbol{x} \in \mathbb{R}^{n}$ or $\boldsymbol{x} \in \mathbb{R}^{n \times 1}$.


### Matrices

An expression for a matrix with $m$ rows and $n$ columns can be written as

$$
\boldsymbol{X} =
\begin{bmatrix}
    x_{11} & x_{12}  & \dots  & x_{1n} \\
    x_{21} & x_{22}  & \dots  & x_{2n} \\
    \vdots & \vdots  & \ddots & \vdots \\
    x_{m1} & x_{m2}  & \dots  & x_{mn}
\end{bmatrix},
$$

Here, $x_{ij}$ is the element in row $i$ and column $j$ in the matrix $\boldsymbol{X}$ ($1 \leq i \leq m, 1 \leq j \leq n$). To express that $\boldsymbol{X}$ is a matrix with $m$ rows and $n$ columns consisting of elements from the set of real numbers, we write $\boldsymbol{X} \in \mathbb{R}^{m \times n}$. It is not difficult to see that vectors are a special class of matrices.


### Operations

Assume the elements in the $n$-dimensional vector $\boldsymbol{a}$ are $a_1, \ldots, a_n$, and the elements in the $n$-dimensional vector $\boldsymbol{b}$ are $b_1, \ldots, b_n$. The dot product (internal product) of vectors $\boldsymbol{a}$ and $\boldsymbol{b}$ is a scalar:

$$
\boldsymbol{a} \cdot \boldsymbol{b} = a_1 b_1 + \ldots + a_n b_n.
$$


Assume two matrices with $m$ rows and $n$ columns:

$$
\boldsymbol{A} =
\begin{bmatrix}
    a_{11} & a_{12} & \dots  & a_{1n} \\
    a_{21} & a_{22} & \dots  & a_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1} & a_{m2} & \dots  & a_{mn}
\end{bmatrix},\quad
\boldsymbol{B} =
\begin{bmatrix}
    b_{11} & b_{12} & \dots  & b_{1n} \\
    b_{21} & b_{22} & \dots  & b_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    b_{m1} & b_{m2} & \dots  & b_{mn}
\end{bmatrix}.
$$

The transpose of a matrix $\boldsymbol{A}$ with $m$ rows and $n$ columns is a matrix with $n$ rows and $m$ columns whose rows are formed from the columns of the original matrix:

$$
\boldsymbol{A}^\top =
\begin{bmatrix}
    a_{11} & a_{21} & \dots  & a_{m1} \\
    a_{12} & a_{22} & \dots  & a_{m2} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{1n} & a_{2n} & \dots  & a_{mn}
\end{bmatrix}.
$$


To add two matrices of the same shape, we add them element-wise:

$$
\boldsymbol{A} + \boldsymbol{B} =
\begin{bmatrix}
    a_{11} + b_{11} & a_{12} + b_{12} & \dots  & a_{1n} + b_{1n} \\
    a_{21} + b_{21} & a_{22} + b_{22} & \dots  & a_{2n} + b_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1} + b_{m1} & a_{m2} + b_{m2} & \dots  & a_{mn} + b_{mn}
\end{bmatrix}.
$$

We use the symbol $\odot$ to indicate the element-wise multiplication of two matrices:

$$
\boldsymbol{A} \odot \boldsymbol{B} =
\begin{bmatrix}
    a_{11}  b_{11} & a_{12}  b_{12} & \dots  & a_{1n}  b_{1n} \\
    a_{21}  b_{21} & a_{22}  b_{22} & \dots  & a_{2n}  b_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1}  b_{m1} & a_{m2}  b_{m2} & \dots  & a_{mn}  b_{mn}
\end{bmatrix}.
$$

Define a scalar $k$. Multiplication of scalars and matrices is also an element-wise multiplication:


$$
k\boldsymbol{A} =
\begin{bmatrix}
    ka_{11} & ka_{12} & \dots  & ka_{1n} \\
    ka_{21} & ka_{22} & \dots  & ka_{2n} \\
    \vdots & \vdots   & \ddots & \vdots \\
    ka_{m1} & ka_{m2} & \dots  & ka_{mn}
\end{bmatrix}.
$$

Other operations such as scalar and matrix addition, and division by an element are similar to the multiplication operation in the above equation. Calculating the square root or taking logarithms of a matrix are performed by calculating the square root or logarithm, respectively, of each element of the matrix to obtain a matrix with the same shape as the original matrix.

Matrix multiplication is different from element-wise matrix multiplication. Assume $\boldsymbol{A}$ is a matrix with $m$ rows and $p$ columns and $\boldsymbol{B}$ is a matrix with $p$ rows and $n$ columns. The product (matrix multiplication) of these two matrices is denoted

$$
\boldsymbol{A} \boldsymbol{B} =
\begin{bmatrix}
    a_{11} & a_{12} & \dots  & a_{1p} \\
    a_{21} & a_{22} & \dots  & a_{2p} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{i1} & a_{i2} & \dots  & a_{ip} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1} & a_{m2} & \dots  & a_{mp}
\end{bmatrix}
\begin{bmatrix}
    b_{11} & b_{12} & \dots  & b_{1j} & \dots & b_{1n} \\
    b_{21} & b_{22} & \dots  & b_{2j} & \dots  & b_{2n} \\
    \vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\
    b_{p1} & b_{p2} & \dots  & b_{pj} & \dots  & b_{pn}
\end{bmatrix},
$$

is a matrix with $m$ rows and $n$ columns, with the element in row $i$ and column $j$ ($1 \leq i \leq m, 1 \leq j \leq n$) equal to

$$a_{i1}b_{1j}  + a_{i2}b_{2j} + \ldots + a_{ip}b_{pj} = \sum_{k=1}^p a_{ik}b_{kj}.$$


### Norms

Assume the elements in the $n$-dimensional vector $\boldsymbol{x}$ are $x_1, \ldots, x_n$. The $L_p$ norm of the vector $\boldsymbol{x}$ is

$$\|\boldsymbol{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.$$

For example, the $L_1$ norm of $\boldsymbol{x}$ is the sum of the absolute values ​​of the vector elements:

$$\|\boldsymbol{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.$$

While the $L_2$ norm of $\boldsymbol{x}$ is the square root of the sum of the squares of the vector elements:

$$\|\boldsymbol{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2}.$$

We usually use $\|\boldsymbol{x}\|$ to refer to the $L_2$ norm of $\boldsymbol{x}$.

Assume $\boldsymbol{X}$ is a matrix with $m$ rows and $n$ columns. The Frobenius norm of matrix $\boldsymbol{X}$ is the square root of the sum of the squares of the matrix elements:

$$\|\boldsymbol{X}\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2},$$

Here, $x_{ij}$ is the element of matrix $\boldsymbol{X}$ in row $i$ and column $j$.


### Eigenvectors and Eigenvalues


Let $\boldsymbol{A}$ be a matrix with $n$ rows and $n$ columns. If $\lambda$ is a scalar and $\boldsymbol{v}$ is a non-zero $n$-dimensional vector with

$$\boldsymbol{A} \boldsymbol{v} = \lambda \boldsymbol{v},$$

then $\boldsymbol{v}$ is an called eigenvector vector of matrix $\boldsymbol{A}$, and $\lambda$ is called an eigenvalue of $\boldsymbol{A}$ corresponding to $\boldsymbol{v}$.



## Differentials

Here we briefly introduce some basic concepts and calculations for differentials.


### Derivatives and Differentials

Assume the input and output of function $f: \mathbb{R} \rightarrow \mathbb{R}$ are both scalars. The derivative $f$ is defined as

$$f'(x) = \lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h},$$

when the limit exists (and $f$ is said to be differentiable). Given $y = f(x)$, where $x$ and $y$ are the arguments and dependent variables of function $f$, respectively, the following derivative and differential expressions are equivalent:

$$f'(x) = y' = \frac{\text{d}y}{\text{d}x} = \frac{\text{d}f}{\text{d}x} = \frac{\text{d}}{\text{d}x} f(x) = \text{D}f(x) = \text{D}_x f(x),$$

Here, the symbols $\text{D}$ and $\text{d}/\text{d}x$ are also called differential operators. Common differential calculations are $\text{D}C = 0$ ($C$ is a constant), $\text{D}x^n = nx^{n-1}$ ($n$ is a constant), and $\text{D}e^x = e^x$, $\text{D}\ln(x) = 1/x$.

If functions $f$ and $g$ are both differentiable and $C$ is a constant, then

$$
\begin{aligned}
\frac{\text{d}}{\text{d}x} [Cf(x)] &= C \frac{\text{d}}{\text{d}x} f(x),\\
\frac{\text{d}}{\text{d}x} [f(x) + g(x)] &= \frac{\text{d}}{\text{d}x} f(x) + \frac{\text{d}}{\text{d}x} g(x),\\
\frac{\text{d}}{\text{d}x} [f(x)g(x)] &= f(x) \frac{\text{d}}{\text{d}x} [g(x)] + g(x) \frac{\text{d}}{\text{d}x} [f(x)],\\
\frac{\text{d}}{\text{d}x} \left[\frac{f(x)}{g(x)}\right] &= \frac{g(x) \frac{\text{d}}{\text{d}x} [f(x)] - f(x) \frac{\text{d}}{\text{d}x} [g(x)]}{[g(x)]^2}.
\end{aligned}
$$


If functions $y=f(u)$ and $u=g(x)$ are both differentiable, then the Chain Rule states that

$$\frac{\text{d}y}{\text{d}x} = \frac{\text{d}y}{\text{d}u} \frac{\text{d}u}{\text{d}x}.$$


### Taylor Expansion

The Taylor expansion of function $f$ is given by the infinite sum

$$f(x) = \sum_{n=0}^\infty \frac{f^{(n)}(a)}{n!} (x-a)^n,$$

when it exists. Here, $f^{(n)}$ is the $n$th derivative of $f$, and $n!$ is the factorial of $n$. For a sufficiently small number $\epsilon$, we can replace $x$ and $a$ with $x+\epsilon$ and $x$ respectively to obtain

$$f(x + \epsilon) \approx f(x) + f'(x) \epsilon + \mathcal{O}(\epsilon^2).$$

Because $\epsilon$ is sufficiently small, the above formula can be simplified to

$$f(x + \epsilon) \approx f(x) + f'(x) \epsilon.$$



### Partial Derivatives

Let $u = f(x_1, x_2, \ldots, x_n)$ be a function with $n$ arguments. The partial derivative of $u$ with respect to its $i$th  parameter $x_i$ is

$$ \frac{\partial u}{\partial x_i} = \lim_{h \rightarrow 0} \frac{f(x_1, \ldots, x_{i-1}, x_i+h, x_{i+1}, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}.$$


The following partial derivative expressions are equivalent:

$$\frac{\partial u}{\partial x_i} = \frac{\partial f}{\partial x_i} = f_{x_i} = f_i = D_i f = D_{x_i} f.$$

To calculate $\partial u/\partial x_i$, we simply treat $x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_n$ as constants and calculate the derivative of $u$ with respect to $x_i$.



### Gradients


Assume the input of function $f: \mathbb{R}^n \rightarrow \mathbb{R}$ is an $n$-dimensional vector $\boldsymbol{x} = [x_1, x_2, \ldots, x_n]^\top$ and the output is a scalar. The gradient of function $f(\boldsymbol{x})$ with respect to $\boldsymbol{x}$ is a vector of $n$ partial derivatives:

$$\nabla_{\boldsymbol{x}} f(\boldsymbol{x}) = \bigg[\frac{\partial f(\boldsymbol{x})}{\partial x_1}, \frac{\partial f(\boldsymbol{x})}{\partial x_2}, \ldots, \frac{\partial f(\boldsymbol{x})}{\partial x_n}\bigg]^\top.$$


To be concise, we sometimes use $\nabla f(\boldsymbol{x})$ to replace $\nabla_{\boldsymbol{x}} f(\boldsymbol{x})$.

If $\boldsymbol{A}$ is a matrix with $m$ rows and $n$ columns, and $\boldsymbol{x}$ is an $n$-dimensional vector, the following identities hold:

$$
\begin{aligned}
\nabla_{\boldsymbol{x}} \boldsymbol{A} \boldsymbol{x} &= \boldsymbol{A}^\top, \\
\nabla_{\boldsymbol{x}} \boldsymbol{x}^\top \boldsymbol{A}  &= \boldsymbol{A}, \\
\nabla_{\boldsymbol{x}} \boldsymbol{x}^\top \boldsymbol{A} \boldsymbol{x}  &= (\boldsymbol{A} + \boldsymbol{A}^\top)\boldsymbol{x},\\
\nabla_{\boldsymbol{x}} \|\boldsymbol{x} \|^2 &= \nabla_{\boldsymbol{x}} \boldsymbol{x}^\top \boldsymbol{x} = 2\boldsymbol{x}.
\end{aligned}
$$

Similarly if $\boldsymbol{X}$ is a matrix, the
$$\nabla_{\boldsymbol{X}} \|\boldsymbol{X} \|_F^2 = 2\boldsymbol{X}.$$




### Hessian Matrices

Assume the input of function $f: \mathbb{R}^n \rightarrow \mathbb{R}$ is an $n$-dimensional vector $\boldsymbol{x} = [x_1, x_2, \ldots, x_n]^\top$ and the output is a scalar. If all second-order partial derivatives of function $f$ exist and are continuous, then the Hessian matrix $\boldsymbol{H}$ of $f$ is a matrix with $m$ rows and $n$ columns given by

$$
\boldsymbol{H} =
\begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \dots  & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \dots  & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \dots  & \frac{\partial^2 f}{\partial x_n^2}
\end{bmatrix},
$$

Here, the second-order partial derivative is evaluated

$$\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial }{\partial x_i} \left(\frac{\partial f}{ \partial x_j}\right).$$



## Probability

Finally, we will briefly introduce conditional probability, expectation, and uniform distribution.

### Conditional Probability

Denote the probability of event $A$ and event $B$ as $\mathbb{P}(A)$ and $\mathbb{P}(B)$, respectively. The probability of the simultaneous occurrence of the two events is denoted as $\mathbb{P}(A \cap B)$ or $\mathbb{P}(A, B)$. If $B$ has non-zero probability, the conditional probability of event $A$ given that $B$ has occurred is

$$\mathbb{P}(A \mid B) = \frac{\mathbb{P}(A \cap B)}{\mathbb{P}(B)},$$

That is,

$$\mathbb{P}(A \cap B) = \mathbb{P}(B) \mathbb{P}(A \mid B) = \mathbb{P}(A) \mathbb{P}(B \mid A).$$

If

$$\mathbb{P}(A \cap B) = \mathbb{P}(A) \mathbb{P}(B),$$

then $A$ and $B$ are said to be independent of each other.


### Expectation

A random variable takes values that represent possible outcomes of an experiment. The expectation (or average) of the random variable $X$ is denoted as

$$\mathbb{E}(X) = \sum_{x} x \mathbb{P}(X = x).$$


### Uniform Distribution

Assume random variable $X$ obeys a uniform distribution over $[a, b]$, i.e. $X \sim U( a, b)$. In this case, random variable $X$ has the same probability of being any number between $a$ and $b$.


## Summary

* This section summarizes the basic knowledge of linear algebra, differentiation, and probability required to understand the contents in this book.


## Exercise

* Find the gradient of function $f(\boldsymbol{x}) = 3x_1^2 + 5e^{x_2}$.

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2397)

![](../img/qr_math.svg)
