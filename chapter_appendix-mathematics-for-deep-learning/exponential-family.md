# Exponential Family
:label:`sec_exp_family`


Constructing and optimizing the right objective function is crucial in deep learning. We normally use methods such as maximum likelihood estimation (:numref:`sec_maximum_likelihood`) to optimize the objective function and find the "best" set of weights. As we may allude in :numref:`sec_softmax_regression`, a widely used technique to approximate the final output $\mathbf{y}$ is by assuming it follows an exponential family distribution. The objection functions of linear regression in :numref:`sec_linear_regression` and softmax regression in 
:numref:`sec_softmax_regression` are two commonly used examples for  exponential family. So what is the exponential family and why is it so powerful and prevalent in deep learning optimization? In this section, we will dive into the fundamental of exponential family.


The *exponential family* is a set of distributions which can be expressed in 
the below form:

$$\mathbf{x \mid \mathbf{\eta}} \sim exp \big{(} \mathbf{\eta}, T(\cdot) \big{)}$$

where 
- $\mathbf{x} = (x_1, x_2, ..., x_n) \in \mathbb{R}^n$ is a random vector;
- $\mathbf{\eta} = (\eta_1, \eta_2, ..., \eta_l) \in \mathbb{R}^l$ 
is called as *natural parameters* (or *canonical parameters*).


To be more concrete, the probability density function of the exponential 
family can be written in as follows:

$$p(\mathbf{x} | \mathbf{\eta}) = h(\mathbf{x}) \cdot exp \big{(} 
\eta^{\top} \cdot T\mathbf(x) - A(\mathbf{\eta}) \big{)}$$
:eqlabel:`eq_exp_pdf`

where the three functions of $\mathbf{x}$ and $\mathbf{\eta}$ in the 
$f_{\eta}(\mathbf{x})$ are known:
- $T(\mathbf{x})= (T_1(\mathbf{x}), T_2(\mathbf{x}), ..., T_l(\mathbf{x}))$ 
is called as the *sufficient statistics* for $\eta$. That is, the 
information represented by $T(\mathbf{x})$ is sufficient to calculate the 
parameter $\eta$, no other information from the sample $\mathbf{x}$'s is 
needed for estimating
$\eta$. For example, the sample mean is the sufficient statistics for the 
true mean parameter ($\eta$) of a normal distribution with known variance;
- $h(\mathbf{x})$ is known as the *underlying measure* or the *base measure*;
- $A(\mathbf{\eta})$ is referred to as the *cumulant function*, which ensures 
that the above distribution :eqref:`eq_exp_pdf` integrates to one, i.e.,

$$  A(\mathbf{\eta}) = \log \int h(\mathbf{x}) \cdot exp \big{(} 
\eta^{\top} \cdot T\mathbf(x) \big{)} dx.$$

## Examples of Exponential Family

Recall some widely used distributions in :numref:`sec_distributions`, let's
demonstrate their ties of blood to the exponential family. Supposed we have a 
random variable $X$ for the below examples. What is more, for the sake of simplicity, we 
assume $X$ is a univariate variable (i.e., $X$ only has one dimension).

### Bernoulli

As we have seen in :numref:`sec_distributions`, if $X$ follows a Bernoulli 
distribution with a probability $p$ (i.e., $X \sim \mathrm{Bernoulli}(p)$), 
then the probability density function of $X$ is:

$$
\begin{aligned}
p(x | p) &= p^x \cdot (1-p)^{1-x} \\
&= exp \big{\{} \log \big{(} p^x \cdot (1-p)^{1-x} \big{)} \\
&= exp \big{\{} x \log p + (1-x) \log (1-p) \big{\}}\\
&= exp \big{\{} x \log \frac{p}{1-p} - (-\log(1-p)) \big{\}}.\\
\end{aligned}
$$

Hence, the parameters of the exponential family format in :eqref:`eq_exp_pdf` can be explained as:

* $\eta = \log \frac{p}{1-p}$, 
* $T(x) = x$,
* $h(x) = 1$,
* and $A(\eta) = -\log(1-p) = \log(1+e^\eta)$.


### Poisson

If $X$ follows a Poisson distribution (in :numref:`sec_distributions`) with a rate or shape parameter $\lambda$ (i.e., $X \sim \mathrm{Poisson}(\lambda)$), then the probability density function of $X$ is:

$$
\begin{aligned}
p(x | \lambda) &= \frac{\lambda^x e^{-\lambda}}{x!} \\
&= \frac{1}{x!} exp\big{\{} x \log \lambda -\lambda \big{\}}.\\
\end{aligned}
$$


Hence, the parameters in :eqref:`eq_exp_pdf` can be explained as:

* $\eta = \log \lambda$, 
* $T(x) = x$,
* $h(x) = \frac{1}{x!}$,
* and $A(\eta) = \lambda = e^{\eta}$.


### Gaussian

From :numref:`sec_distributions`, if $X$ follows a Gaussian distribution with mean $\mu$ and standard deviation $\sigma$ (i.e., $X \sim \mathcal{N}(\mu, \sigma^2)$), then the probability density function of $X$ is:

$$
\begin{aligned}
p(x | \mu, \sigma) &= \frac{1}{\sqrt{2 \pi \sigma^2}} exp \big{\{} \frac{-(x-\mu)^2}{2 \sigma^2} \big{\}} \\
&= \frac{1}{\sqrt{2 \pi}} \cdot exp \big{\{} \frac{\mu}{\sigma^2}x - \frac{1}{2 \sigma^2} x^2 - \big{(} \log(\sigma) + \frac{1}{2 \sigma^2} \mu^2 \big{)} \big{\}} .
\end{aligned}
$$

Hence, the parameters in :eqref:`eq_exp_pdf` can be explained as:


* $\eta = \begin{bmatrix} \eta_1 \\ \eta_2 \end{bmatrix} = \begin{bmatrix} \frac{\mu}{\sigma^2} \\ -\frac{1}{2 \sigma^2}  \end{bmatrix}$,
* $T(x) = \begin{bmatrix}x\\x^2\end{bmatrix}$,
* $h(x) = \frac{1}{\sqrt{2 \pi}}$,
* and $A(\eta) = \log(\sigma) + \frac{1}{2 \sigma^2} \mu^2 = -\frac{\eta_1^2}{4 \eta_2} - \frac{\log(-2 \eta_2)}{2}$.


Besides the above distributions, Binomial, Multinomial, Exponential and other commonly seen distributions are members of exponential family as well.

## The Gradients

Optimizing a neural network requires to calculate the gradients of the objective functions.

As we have seen from the above examples, it seems that exponential family is just a generalized model which covers a lot of widely used statistics distributions. However, if we dive deep to the gradients of the exponential family, we may magically discover that the gradients can be calculated by calculate its mean and variance of the exponential family distribution. To be more concrete, the first order derivative of the cumulant function $A(\eta)$ can be obtained by the mean $\mathrm{E} [T(X)]$, and second order derivative of $A(\eta)$  can be represented by the variance $\mathrm{Var} [T(X)]$. Now, let's unveil the black box piece by piece!

Let's start with the first order derivative of $A(\eta)$ of a exponential family distribution:


$$
\begin{aligned}
\nabla_{\mathbf{\eta^{\top}}} A(\mathbf{\eta}) &= \nabla_{\mathbf{\eta}^{\top}} \log \int h(\mathbf{x}) \cdot exp \big{(} \eta^{\top} \cdot T\mathbf(x) \big{)} dx \\
&= \frac{\int T\mathbf(x) \cdot h(\mathbf{x}) \cdot exp \big{(} \eta^{\top} \cdot T\mathbf(x) \big{)} dx}{\int h(\mathbf{x}) \cdot exp \big{(} \eta^{\top} \cdot T\mathbf(x) \big{)} dx}. \\
\end{aligned}
$$

As the denominator $\int h(\mathbf{x}) \cdot exp \big{(} \eta^{\top} \cdot T\mathbf(x) \big{)} dx = e^{- A(\mathbf{\eta})}$,

$$
\begin{aligned}
\nabla_{\mathbf{\eta^{\top}}} A(\mathbf{\eta})
&= \int T\mathbf(x) \cdot h(\mathbf{x}) \cdot exp \big{(} \eta^{\top} \cdot T\mathbf(x) - A(\mathbf{\eta}) \big{)} dx.\\
\end{aligned}
$$

If we play some math tricks using the *moment generating function*, the last above equation is just the mean of the sufficient statistics
$T\mathbf(x)$, i.e.,

$$
\begin{aligned}
\nabla_{\mathbf{\eta^{\top}}} A(\mathbf{\eta})
&= \mathrm{E} [T(X)].\\
\end{aligned}
$$

With this surprise finding, it is much easier to calculate the gradients of objective function by simply calculating the mean of the sufficient statistics. Is this math serendipity super amazing?! If you would like to a detailed application using the above trick, check :numref:`subsec_softmax_and_derivatives` to get more inspiration!

## Summary

* Exponential family plays a crucial role in constructing objective functions of deep learning.
* The objective functions of both linear regression and softmax regression belong to exponential family.
* To obtain the gradients of the cumulant function of exponential family distribution, we can simply calculate its sufficient statistics.


## Exercises

1. Can you try to list other examples of exponential family from its mathematical definition?
1. We mentioned that the second derivative of the cumulant function can be calculated by the variance of sufficient statistics. Can you derive a rigorous math proof? (Hint: apply the similar math tricks as we calculate the first order derivative.)
