# Introduction to Gaussian processes

Understanding Gaussian processes (GPs) is important for reasoning about model construction and generalization, and for achieving state-of-the-art performance in a variety of applications, including active learning, and hyperparameter tuning in deep learning. GPs are everywhere, and it is in our interests to know what they are and how we can use them. 

## Definition

A Gaussian process is defined as _a collection of random variables, any finite number of which have a joint Gaussian distribution_. If a function $f(x)$ is a Gaussian process, with _mean function_ $m(x)$ and _covariance function_ or _kernel_ $k(x,x')$, $f(x) \sim \mathcal{GP}(m, k)$, then any collection of function values queried at any collection of input points $x$ (times, spatial locations, image pixels, etc.), has a joint multivariate Gaussian distribution with mean vector $\mu$ and covariance matrix $K$: $f(x_1),\dots,f(x_n) \sim \mathcal{N}(\mu, K)$, where $\mu_i = \mathbb{E}[f(x_i)] = m(x_i)$ and $K_{ij} = cov(f(x_i),f(x_j)) = k(x_i,x_j)$. 

This definition may seem abstract and inaccessible, but Gaussian processes are in fact very simple objects. Any function

$f(x) = w^{\top} \phi(x) = \langle w, \phi(x) \rangle$, (1) 

with $w$ drawn from a Gaussian (normal) distribution, and $\phi$ being any vector of basis functions, for example $\phi(x) = (1, x, x^2, ..., x^d)^{\top}$, 
is a Gaussian process. Moreover, any Gaussian process f(x) can be expressed in the form of equation (1). Let's consider a few concrete examples, to begin getting acquainted with Gaussian processes, after which we can appreciate how simple and useful they really are.

## A simple Gaussian process

Suppose 

$f(x) = w_0 + w_1 x$, and $w_0, w_1 \sim \mathcal{N}(0,1)$, with $w_0, w_1, x$ all in one dimension. 

We can equivalently write this function as the inner product $f(x) = (w_0, w_1)(1, x)^{\top}$. In equation (1) above, $w = (w_0, w_1)^{\top}$ and $\phi(x) = (1,x)^{\top}$. 

For any $x$, $f(x)$ is a sum of two Gaussian random variables. Since Gaussians are closed under addition, $f(x)$ is also a Gaussian random variable for any $x$. In fact, we can compute for any particular $x$ that $f(x)$ is $\mathcal{N}(0,1+x^2)$. Similarly, the joint distribution for any collection of function values, $(f(x_1),\dots,f(x_n))$, for any collection of inputs $x_1,\dots,x_n$, is a multivariate Gaussian distribution. Therefore $f(x)$ is a Gaussian process. 

In short, $f(x)$ is a _random function_, or a _distribution over functions_. We can gain some insights into this distribution by repeatedly sampling values for $w_0, w_1$, and visualizing the corresponding functions $f(x)$, which are straight lines with slopes and different intercepts, as follows:

(! Add plot here).

If we change the distribution over $w_0, w_1$ to $\mathcal{N}(0,\alpha^2)$, how do you imagine varying $\alpha$ affects the distribution over functions?

## From weight space to function space 

In the plot above, we saw how a distribution over parameters in a model induces a distribution over functions. While we often have ideas about the functions we want to model --- whether they're smooth, periodic, quickly varying, etc. --- it's relatively tedious to reason about the parameters, which are largely uninterpretable.




