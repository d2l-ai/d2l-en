# Notation
:label:`chap_notation`

Throughout this book, we adhere 
to the following notational conventions.
Note that some of these symbols are placeholders,
while others refer to specific objects.
As a general rule of thumb, 
the indefinite article "a" often indicates
that the symbol is a placeholder
and that similarly formatted symbols
can denote other objects of the same type.
For example, "$x$: a scalar" means 
that lowercased letters generally
represent scalar values,
but "$\mathbb{Z}$: the set of integers"
refers specifically to the symbol $\mathbb{Z}$.



## Numerical Objects

* $x$: a scalar
* $\mathbf{x}$: a vector
* $\mathbf{X}$: a matrix
* $\mathsf{X}$: a general tensor
* $\mathbf{I}$: the identity matrix (of some given dimension), i.e., a square matrix with $1$ on all diagonal entries and $0$ on all off-diagonals
* $x_i$, $[\mathbf{x}]_i$: the $i^\mathrm{th}$ element of vector $\mathbf{x}$
* $x_{ij}$, $x_{i,j}$,$[\mathbf{X}]_{ij}$, $[\mathbf{X}]_{i,j}$: the element of matrix $\mathbf{X}$ at row $i$ and column $j$.



## Set Theory


* $\mathcal{X}$: a set
* $\mathbb{Z}$: the set of integers
* $\mathbb{Z}^+$: the set of positive integers
* $\mathbb{R}$: the set of real numbers
* $\mathbb{R}^n$: the set of $n$-dimensional vectors of real numbers
* $\mathbb{R}^{a\times b}$: The set of matrices of real numbers with $a$ rows and $b$ columns
* $|\mathcal{X}|$: cardinality (number of elements) of set $\mathcal{X}$
* $\mathcal{A}\cup\mathcal{B}$: union of sets $\mathcal{A}$ and $\mathcal{B}$
* $\mathcal{A}\cap\mathcal{B}$: intersection of sets $\mathcal{A}$ and $\mathcal{B}$
* $\mathcal{A}\setminus\mathcal{B}$: set subtraction of $\mathcal{B}$ from $\mathcal{A}$ (contains only those elements of $\mathcal{A}$ that do not belong to $\mathcal{B}$)



## Functions and Operators


* $f(\cdot)$: a function
* $\log(\cdot)$: the natural logarithm (base $e$)
* $\log_2(\cdot)$: logarithm with base $2$
* $\exp(\cdot)$: the exponential function
* $\mathbf{1}(\cdot)$: the indicator function, evaluates to $1$ if the boolean argument is true and $0$ otherwise
* $\mathbf{1}_{\mathcal{X}}(z)$: the set-membership indicator function, evaluates to $1$ if the element $z$ belongs to the set $\mathcal{X}$ and $0$ otherwise
* $\mathbf{(\cdot)}^\top$: transpose of a vector or a matrix
* $\mathbf{X}^{-1}$: inverse of matrix $\mathbf{X}$
* $\odot$: Hadamard (elementwise) product
* $[\cdot, \cdot]$: concatenation
* $\|\cdot\|_p$: $\ell_p$ norm
* $\|\cdot\|$: $\ell_2$ norm
* $\langle \mathbf{x}, \mathbf{y} \rangle$: dot product of vectors $\mathbf{x}$ and $\mathbf{y}$
* $\sum$: summation over a collection of elements
* $\prod$: product over a collection of elements
* $\stackrel{\mathrm{def}}{=}$: an equality asserted as a definition of the symbol on the left-hand side



## Calculus

* $\frac{dy}{dx}$: derivative of $y$ with respect to $x$
* $\frac{\partial y}{\partial x}$: partial derivative of $y$ with respect to $x$
* $\nabla_{\mathbf{x}} y$: gradient of $y$ with respect to $\mathbf{x}$
* $\int_a^b f(x) \;dx$: definite integral of $f$ from $a$ to $b$ with respect to $x$
* $\int f(x) \;dx$: indefinite integral of $f$ with respect to $x$



## Probability and Information Theory

* $X$: a random variable
* $P$: a probability distribution
* $X \sim P$: the random variable $X$ follows distribution $P$
* $P(X=x)$: the probability assigned to the event where random variable $X$ takes value $x$
* $P(X \mid Y)$: the conditional probability distribution of $X$ given $Y$
* $p(\cdot)$: a probability density function (PDF) associated with distribution P
* ${E}[X]$: expectation of a random variable $X$
* $X \perp Y$: random variables $X$ and $Y$ are independent
* $X \perp Y \mid Z$: random variables  $X$  and  $Y$ are conditionally independent given $Z$
* $\sigma_X$: standard deviation of random variable $X$
* $\mathrm{Var}(X)$: variance of random variable $X$, equal to $\sigma^2_X$
* $\mathrm{Cov}(X, Y)$: covariance of random variables $X$ and $Y$
* $\rho(X, Y)$: the Pearson correlation coefficient between $X$ and $Y$, equals $\frac{\mathrm{Cov}(X, Y)}{\sigma_X \sigma_Y}$
* $H(X)$: entropy of random variable $X$
* $D_{\mathrm{KL}}(P\|Q)$: the KL-divergence (or relative entropy) from distribution $Q$ to distribution $P$



[Discussions](https://discuss.d2l.ai/t/25)
